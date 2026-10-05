"""
Detect, cross-check, and evaluate anomalies in unseen LIGO strain data.

This module applies the trained convolutional autoencoder to previously unseen
LIGO strain data. Candidate anomalies are identified using detector-specific
reconstruction-error thresholds and cross-checked between the Livingston (L1)
and Hanford (H1) detectors.

Candidates appearing in both detectors undergo fine-time correlation analysis
and comparison against quality-filtered background correlations. Candidates
that survive these validation stages are characterized using calibrated strain
and passed to the physics analysis module.

The empirical p-value used here is a project-level screening statistic and is
not equivalent to the formal significance methods used by LIGO.
"""

import h5py
import torch

from pycbc.filter import highpass, lowpass
from pycbc.types import TimeSeries

from model import StrainAutoencoder, normalize_windows
from physics import analyze_candidate_physics


# Strain data settings
SAMPLE_RATE = 4096

# Background data used to establish detector-specific anomaly thresholds
L1_BACKGROUND_FILES = [
    "data/L-L1_LOSC_4_V1-1126068224-4096.hdf5",
    "data/L-L1_LOSC_4_V1-1126072320-4096.hdf5",
]

H1_BACKGROUND_FILES = [
    "data/H-H1_LOSC_4_V1-1126072320-4096.hdf5",
]

# Previously unseen data used for anomaly detection
L1_TEST_FILE = "data/L-L1_LOSC_4_V1-1126076416-4096.hdf5"
H1_TEST_FILE = "data/H-H1_LOSC_4_V1-1126076416-4096.hdf5"
TEST_GPS_START = 1126076416

# Anomaly-detection and validation settings
ANOMALY_PERCENTILE = 0.99
BACKGROUND_COMPARISONS = 200
SIGNIFICANCE_THRESHOLD = 0.05
MAX_PHYSICAL_DELAY_MS = 10.0


"""
    Determine which seconds pass all quality and hardware-injection checks.

    A second is accepted only when every available data-quality flag and every
    no-hardware-injection flag are set.

    Args:
        file: Open LIGO HDF5 file.

    Returns:
        array: Boolean mask indicating which seconds pass all checks.
"""
def get_quality_mask(file):
    dq_mask = file["quality/simple/DQmask"][:]
    dq_names = file["quality/simple/DQShortnames"][:]

    injection_mask = file["quality/injections/Injmask"][:]
    injection_names = file["quality/injections/InjShortnames"][:]

    required_dq_mask = (1 << len(dq_names)) - 1
    required_injection_mask = (1 << len(injection_names)) - 1

    quality_pass = (dq_mask & required_dq_mask) == required_dq_mask
    no_injection = (
        injection_mask & required_injection_mask
    ) == required_injection_mask

    return quality_pass & no_injection


"""
    Load one-second strain windows that pass all quality checks.

    Each passing second becomes one 4096-sample window. When the GPS start
    time is provided, the corresponding GPS second is recorded for each
    accepted window.

    Args:
        file_path: Path to a LIGO HDF5 strain file.
        gps_start: GPS start time of the file, if GPS timestamps are needed.

    Returns:
        tuple: Passing strain windows and their corresponding GPS times.
"""
def get_passing_windows(file_path, gps_start=None):
    windows = []
    gps_times = []

    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][:]
        passing_seconds = get_quality_mask(file)

        for second, passes in enumerate(passing_seconds):
            if passes:
                start = second * SAMPLE_RATE
                end = start + SAMPLE_RATE
                windows.append(strain[start:end])

                if gps_start is not None:
                    gps_times.append(gps_start + second)

    return windows, gps_times


"""
    Collect quality-passing background windows from multiple LIGO files.

    Args:
        file_paths: Collection of background HDF5 file paths.

    Returns:
        list: Combined one-second strain windows from all supplied files.
"""
def get_background_windows(file_paths):
    windows = []

    for file_path in file_paths:
        file_windows, _ = get_passing_windows(file_path)
        windows.extend(file_windows)

    return windows


"""
    Convert strain windows into normalized PyTorch tensors.

    Args:
        windows: Collection of one-second strain windows.

    Returns:
        Tensor: Normalized data with shape
        (number of windows, 1, samples per window).
"""
def prepare_data(windows):
    data = torch.stack([
        torch.tensor(window, dtype=torch.float64)
        for window in windows
    ])

    return normalize_windows(data.unsqueeze(1))


"""
    Calculate autoencoder reconstruction error for each strain window.

    Mean squared error between each input and its reconstruction provides the
    anomaly score used by the detection pipeline.

    Args:
        model: Trained StrainAutoencoder.
        data: Normalized strain windows.

    Returns:
        Tensor: One reconstruction-error score per window.
"""
def calculate_reconstruction_errors(model, data):
    model.eval()

    with torch.no_grad():
        reconstructed = model(data)
        return torch.mean((data - reconstructed) ** 2, dim=(1, 2))


"""
    Calculate a detector-specific anomaly threshold from background strain.

    The threshold is the configured percentile of reconstruction errors
    observed in quality-filtered background data.

    Args:
        model: Trained StrainAutoencoder.
        file_paths: Background files for one detector.

    Returns:
        tuple: Reconstruction-error threshold and number of background windows.
"""
def calculate_threshold(model, file_paths):
    windows = get_background_windows(file_paths)
    data = prepare_data(windows)
    errors = calculate_reconstruction_errors(model, data)

    threshold = torch.quantile(errors, ANOMALY_PERCENTILE).item()

    return threshold, len(windows)


"""
    Load, band-limit, and standardize a section of LIGO strain.

    The strain is restricted to approximately 30-500 Hz before being
    standardized. This representation is used for correlation analysis rather
    than physical amplitude measurements.

    Args:
        file_path: Path to a LIGO HDF5 strain file.
        start_sample: First strain sample to load.
        number_of_samples: Number of samples to load.

    Returns:
        Tensor: Conditioned strain with zero mean and unit standard deviation.
"""
def get_conditioned_strain(file_path, start_sample, number_of_samples):
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][
            start_sample:start_sample + number_of_samples
        ]

    strain = TimeSeries(strain, delta_t=1 / SAMPLE_RATE)
    strain = highpass(strain, 30.0)
    strain = lowpass(strain, 500.0)

    strain = torch.tensor(strain.numpy(), dtype=torch.float64)

    return (strain - strain.mean()) / strain.std()


"""
    Find the strongest H1-L1 correlation across small relative time shifts.

    Relative shifts up to 20 milliseconds in either direction are tested.
    Absolute correlation strength is used because the two detectors can
    respond to the same gravitational-wave signal with opposite signs.

    Args:
        l1: Conditioned Livingston strain.
        h1: Conditioned Hanford strain.

    Returns:
        tuple: Signed best correlation and corresponding delay in milliseconds.
"""
def get_max_correlation(l1, h1):
    max_delay_samples = round(0.020 * SAMPLE_RATE)

    best_correlation = 0.0
    best_delay_samples = 0

    for delay in range(-max_delay_samples, max_delay_samples + 1):
        if delay < 0:
            l1_section = l1[-delay:]
            h1_section = h1[:delay]
        elif delay > 0:
            l1_section = l1[:-delay]
            h1_section = h1[delay:]
        else:
            l1_section = l1
            h1_section = h1

        correlation = torch.mean(l1_section * h1_section).item()

        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_delay_samples = delay

    delay_ms = best_delay_samples / SAMPLE_RATE * 1000

    return best_correlation, delay_ms


"""
    Perform fine-time H1-L1 comparison around a candidate.

    Two seconds of conditioned strain beginning at the candidate second are
    compared to determine the strongest cross-detector correlation and its
    relative time delay.

    Args:
        l1_file_path: Livingston test-file path.
        h1_file_path: Hanford test-file path.
        candidate_gps: GPS second of the candidate.
        file_gps_start: GPS start time of the test files.

    Returns:
        tuple: Best delay in milliseconds and signed correlation.
"""
def fine_time_comparison(
    l1_file_path,
    h1_file_path,
    candidate_gps,
    file_gps_start,
):
    print("\nFine-time analysis:")
    print("Candidate GPS:", candidate_gps)

    start_second = candidate_gps - file_gps_start
    start_sample = start_second * SAMPLE_RATE
    number_of_samples = SAMPLE_RATE * 2

    l1 = get_conditioned_strain(
        l1_file_path, start_sample, number_of_samples
    )
    h1 = get_conditioned_strain(
        h1_file_path, start_sample, number_of_samples
    )

    best_correlation, best_delay_ms = get_max_correlation(l1, h1)

    print("Best conditioned H1-L1 delay:", f"{best_delay_ms:.3f} ms")
    print("Conditioned correlation:", f"{best_correlation:.6f}")

    # A real signal cannot propagate between Hanford and Livingston with an
    # inter-detector delay much greater than approximately 10 milliseconds.
    if abs(best_delay_ms) <= MAX_PHYSICAL_DELAY_MS:
        print("Delay is within the approximate physical H1-L1 range.")
    else:
        print("Delay is outside the approximate physical H1-L1 range.")

    return best_delay_ms, best_correlation


"""
    Return which seconds pass all quality and hardware-injection checks.

    Args:
        file_path: Path to a LIGO HDF5 strain file.

    Returns:
        array: Boolean quality mask with one value per second.
"""
def get_passing_seconds(file_path):
    with h5py.File(file_path, "r") as file:
        return get_quality_mask(file)


"""
    Estimate how unusual a candidate correlation is relative to background.

    Quality-passing two-second regions from both detectors are sampled across
    the available background. The maximum H1-L1 correlation for each region
    forms an empirical background distribution.

    The returned p-value is a project-level screening statistic. It should not
    be interpreted as formal LIGO false-alarm probability or astrophysical
    detection significance.

    Args:
        candidate_correlation: Cross-detector correlation of the candidate.

    Returns:
        float or None: Empirical p-value, or None when no valid background
        comparison regions are available.
"""
def calculate_correlation_significance(candidate_correlation):
    print("\nCorrelation significance test...")

    l1_background_file = "data/L-L1_LOSC_4_V1-1126072320-4096.hdf5"
    h1_background_file = "data/H-H1_LOSC_4_V1-1126072320-4096.hdf5"

    l1_passing = get_passing_seconds(l1_background_file)
    h1_passing = get_passing_seconds(h1_background_file)

    common_length = min(len(l1_passing), len(h1_passing))
    l1_passing = l1_passing[:common_length]
    h1_passing = h1_passing[:common_length]

    # Correlation uses two-second regions, so both consecutive seconds must
    # pass the quality requirements in both detectors.
    valid_start_seconds = [
        second
        for second in range(common_length - 1)
        if (
            l1_passing[second]
            and l1_passing[second + 1]
            and h1_passing[second]
            and h1_passing[second + 1]
        )
    ]

    print("Quality-passing paired regions:", len(valid_start_seconds))

    if not valid_start_seconds:
        print("No shared quality-passing background regions available.")
        return None

    # Spread comparisons across the available background instead of selecting
    # only a contiguous portion of the file.
    comparison_count = min(
        BACKGROUND_COMPARISONS,
        len(valid_start_seconds),
    )

    if comparison_count == 1:
        selected_seconds = [valid_start_seconds[0]]
    else:
        selected_seconds = [
            valid_start_seconds[
                round(
                    index
                    * (len(valid_start_seconds) - 1)
                    / (comparison_count - 1)
                )
            ]
            for index in range(comparison_count)
        ]

    number_of_samples = SAMPLE_RATE * 2
    background_correlations = []

    for start_second in selected_seconds:
        start_sample = start_second * SAMPLE_RATE

        l1 = get_conditioned_strain(
            l1_background_file, start_sample, number_of_samples
        )
        h1 = get_conditioned_strain(
            h1_background_file, start_sample, number_of_samples
        )

        correlation, _ = get_max_correlation(l1, h1)
        background_correlations.append(abs(correlation))

    candidate_strength = abs(candidate_correlation)

    stronger_background = sum(
        correlation >= candidate_strength
        for correlation in background_correlations
    )

    # Adding one to the numerator and denominator prevents an empirical
    # p-value of exactly zero with a finite background sample.
    empirical_p_value = (
        stronger_background + 1
    ) / (
        len(background_correlations) + 1
    )

    background_tensor = torch.tensor(
        background_correlations,
        dtype=torch.float64,
    )

    print("Background comparisons:", len(background_correlations))
    print(
        "Median background max correlation:",
        f"{torch.median(background_tensor).item():.6f}",
    )
    print(
        "95th percentile background correlation:",
        f"{torch.quantile(background_tensor, 0.95).item():.6f}",
    )
    print(
        "99th percentile background correlation:",
        f"{torch.quantile(background_tensor, 0.99).item():.6f}",
    )
    print("Candidate correlation:", f"{candidate_strength:.6f}")
    print("Background regions >= candidate:", stronger_background)
    print("Empirical p-value:", f"{empirical_p_value:.6f}")

    return empirical_p_value


"""
    Load calibrated detector strain while preserving its physical scale.

    Unlike the strain used for correlation analysis, this data is not
    standardized because its dimensionless physical amplitude is required for
    later interpretation.

    Args:
        file_path: Path to a LIGO HDF5 strain file.
        start_sample: First strain sample to load.
        number_of_samples: Number of samples to load.

    Returns:
        Tensor: Calibrated strain in its original physical scale.
"""
def get_physical_strain(file_path, start_sample, number_of_samples):
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][
            start_sample:start_sample + number_of_samples
        ]

    return torch.tensor(strain, dtype=torch.float64)


"""
    Band-limit calibrated strain while preserving its physical amplitude.

    Args:
        file_path: Path to a LIGO HDF5 strain file.
        start_sample: First strain sample to load.
        number_of_samples: Number of samples to load.

    Returns:
        Tensor: Approximately 30-500 Hz calibrated strain.
"""
def get_conditioned_physical_strain(
    file_path,
    start_sample,
    number_of_samples,
):
    strain = get_physical_strain(
        file_path, start_sample, number_of_samples
    )

    strain = TimeSeries(strain.numpy(), delta_t=1 / SAMPLE_RATE)
    strain = highpass(strain, 30.0)
    strain = lowpass(strain, 500.0)

    return torch.tensor(strain.numpy(), dtype=torch.float64)


"""
    Measure basic properties of a candidate that survives validation.

    The candidate is characterized using both standardized conditioned strain
    and calibrated physical strain. Its peak amplitude, peak physical strain,
    and dominant frequency are reported.

    Args:
        file_path: Path to the detector strain file.
        candidate_gps: GPS second of the surviving candidate.
        file_gps_start: GPS start time of the strain file.

    Returns:
        dict: Conditioned peak amplitude, physical peak strain, and dominant
        frequency.
"""
def characterize_candidate(file_path, candidate_gps, file_gps_start):
    start_second = candidate_gps - file_gps_start
    start_sample = start_second * SAMPLE_RATE
    number_of_samples = SAMPLE_RATE * 2

    physical_strain = get_conditioned_physical_strain(
        file_path, start_sample, number_of_samples
    )
    peak_physical_strain = torch.max(torch.abs(physical_strain)).item()

    strain = get_conditioned_strain(
        file_path, start_sample, number_of_samples
    )
    peak_amplitude = torch.max(torch.abs(strain)).item()

    frequency_spectrum = torch.fft.rfft(strain)
    frequencies = torch.fft.rfftfreq(
        len(strain),
        d=1 / SAMPLE_RATE,
    )
    power = torch.abs(frequency_spectrum) ** 2

    # Ignore the zero-frequency component when locating the spectral peak.
    dominant_index = torch.argmax(power[1:]).item() + 1
    dominant_frequency = frequencies[dominant_index].item()

    print("\nSignal characterization:")
    print("GPS:", candidate_gps)
    print("Peak conditioned amplitude:", f"{peak_amplitude:.6f}")
    print("Peak physical strain:", f"{peak_physical_strain:.6e}")
    print("Dominant frequency:", f"{dominant_frequency:.2f} Hz")

    return {
        "peak_conditioned_amplitude": peak_amplitude,
        "peak_physical_strain": peak_physical_strain,
        "dominant_frequency_hz": dominant_frequency,
    }


"""
    Run the complete cross-detector anomaly-detection pipeline.

    Detector-specific reconstruction thresholds are calculated from background
    data before unseen L1 and H1 strain is scanned. L1 anomalies are checked
    for nearby H1 anomalies, followed by fine-time correlation and empirical
    background comparison.

    Only candidates that survive every validation stage proceed to physical
    signal characterization and physics analysis.
"""
def main():
    print("Loading trained autoencoder...")

    model = StrainAutoencoder()
    model.load_state_dict(
        torch.load("strain_autoencoder.pth", map_location="cpu")
    )
    model.eval()

    # Detector-specific anomaly thresholds
    print("Calculating L1 background threshold...")
    l1_threshold, l1_background_count = calculate_threshold(
        model, L1_BACKGROUND_FILES
    )
    print(
        f"L1 threshold: {l1_threshold:.6f} "
        f"from {l1_background_count} background windows"
    )

    print("Calculating H1 background threshold...")
    h1_threshold, h1_background_count = calculate_threshold(
        model, H1_BACKGROUND_FILES
    )
    print(
        f"H1 threshold: {h1_threshold:.6f} "
        f"from {h1_background_count} background windows"
    )

    # Scan unseen L1 data for anomalous reconstruction errors.
    print("\nScanning unseen L1 data...")

    l1_windows, l1_gps_times = get_passing_windows(
        L1_TEST_FILE, TEST_GPS_START
    )
    l1_data = prepare_data(l1_windows)
    l1_errors = calculate_reconstruction_errors(model, l1_data)

    l1_candidates = {
        gps_time: l1_errors[index].item()
        for index, gps_time in enumerate(l1_gps_times)
        if l1_errors[index].item() > l1_threshold
    }

    print("L1 candidate anomalies:", len(l1_candidates))

    for gps_time, error in l1_candidates.items():
        print(
            "GPS:",
            gps_time,
            "| L1 reconstruction error:",
            f"{error:.6f}",
        )

    # Scan H1 using its independently calculated anomaly threshold.
    print("\nScanning unseen H1 data...")

    h1_windows, h1_gps_times = get_passing_windows(
        H1_TEST_FILE, TEST_GPS_START
    )
    h1_data = prepare_data(h1_windows)
    h1_errors = calculate_reconstruction_errors(model, h1_data)

    h1_error_by_gps = {
        gps_time: h1_errors[index].item()
        for index, gps_time in enumerate(h1_gps_times)
    }

    # Cross-check each L1 anomaly against the same and adjacent H1 seconds.
    print("\nCross-detector candidate check:")

    confirmed_candidates = []

    for gps_time, l1_error in l1_candidates.items():
        print(f"\nL1 candidate GPS {gps_time}")
        print(f"L1 error: {l1_error:.6f}")

        h1_matches = []

        # One-second autoencoder windows provide only coarse timing, so the
        # neighboring H1 seconds are included in the initial coincidence test.
        for check_time in range(gps_time - 1, gps_time + 2):
            if (
                check_time in h1_error_by_gps
                and h1_error_by_gps[check_time] > h1_threshold
            ):
                h1_matches.append(
                    (check_time, h1_error_by_gps[check_time])
                )

        if h1_matches:
            print("H1 anomaly found nearby:")

            for h1_time, h1_error in h1_matches:
                print(
                    "  GPS:",
                    h1_time,
                    "| H1 reconstruction error:",
                    f"{h1_error:.6f}",
                )

            confirmed_candidates.append(gps_time)
        else:
            print("No H1 anomaly found nearby.")

    print("\nCross-detector candidates:", len(confirmed_candidates))

    for gps_time in confirmed_candidates:
        print("GPS:", gps_time)

    # Fine-time and statistical validation
    surviving_candidates = []

    for gps_time in confirmed_candidates:
        delay_ms, correlation = fine_time_comparison(
            L1_TEST_FILE,
            H1_TEST_FILE,
            gps_time,
            TEST_GPS_START,
        )

        p_value = calculate_correlation_significance(correlation)

        print("\nCandidate classification:")
        print("GPS:", gps_time)

        if abs(delay_ms) > MAX_PHYSICAL_DELAY_MS:
            print("REJECTED")
            print(
                "Reason: H1-L1 delay is outside the physical range."
            )
            continue

        if p_value is None:
            print("REJECTED")
            print(
                "Reason: insufficient valid background "
                "for significance testing."
            )
            continue

        # This p-value threshold is a project-level screening criterion rather
        # than formal LIGO astrophysical detection significance.
        if p_value >= SIGNIFICANCE_THRESHOLD:
            print("REJECTED")
            print(
                "Reason: cross-detector correlation is "
                "compatible with ordinary background."
            )
            continue

        print("PASSED")
        print("Candidate survived cross-detector validation.")

        characterization = characterize_candidate(
            L1_TEST_FILE,
            gps_time,
            TEST_GPS_START,
        )

        physics_result = analyze_candidate_physics(
            characterization["peak_physical_strain"]
        )

        surviving_candidates.append({
            "gps": gps_time,
            "delay_ms": delay_ms,
            "correlation": correlation,
            "p_value": p_value,
            "characterization": characterization,
            "physics": physics_result,
        })

    # Final detection summary
    print("\n========================================")
    print("FINAL DETECTION RESULTS")
    print("========================================")
    print("Initial L1 anomalies:", len(l1_candidates))
    print("Cross-detector candidates:", len(confirmed_candidates))
    print("Surviving candidates:", len(surviving_candidates))

    if surviving_candidates:
        print("\nValidated candidates:")

        for candidate in surviving_candidates:
            print("\nGPS:", candidate["gps"])
            print(
                "H1-L1 delay:",
                f'{candidate["delay_ms"]:.3f} ms',
            )
            print(
                "Correlation:",
                f'{candidate["correlation"]:.6f}',
            )
            print(
                "Empirical p-value:",
                f'{candidate["p_value"]:.6f}',
            )
    else:
        print("\nNo candidates survived validation.")


if __name__ == "__main__":
    main()