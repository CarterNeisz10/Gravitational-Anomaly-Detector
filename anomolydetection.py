"""Detect, cross-check, and evaluate anomalies in unseen LIGO strain data."""

import h5py
import torch

from pycbc.types import TimeSeries
from pycbc.filter import highpass, lowpass

from model import StrainAutoencoder, normalize_windows
from physics import analyze_candidate_physics


SAMPLE_RATE = 4096

L1_BACKGROUND_FILES = [
    "data/L-L1_LOSC_4_V1-1126068224-4096.hdf5",
    "data/L-L1_LOSC_4_V1-1126072320-4096.hdf5",
]

H1_BACKGROUND_FILES = [
    "data/H-H1_LOSC_4_V1-1126072320-4096.hdf5",
]

L1_TEST_FILE = "data/L-L1_LOSC_4_V1-1126076416-4096.hdf5"
H1_TEST_FILE = "data/H-H1_LOSC_4_V1-1126076416-4096.hdf5"

TEST_GPS_START = 1126076416

ANOMALY_PERCENTILE = 0.99
BACKGROUND_COMPARISONS = 200
SIGNIFICANCE_THRESHOLD = 0.05
MAX_PHYSICAL_DELAY_MS = 10.0


def get_passing_windows(file_path, gps_start=None):
    """Load quality-passing strain windows from one LIGO file."""
    windows = []
    gps_times = []

    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][:]

        dq_mask = file["quality/simple/DQmask"][:]
        dq_names = file["quality/simple/DQShortnames"][:]

        injection_mask = file["quality/injections/Injmask"][:]
        injection_names = file["quality/injections/InjShortnames"][:]

        required_dq_mask = (1 << len(dq_names)) - 1
        required_injection_mask = (1 << len(injection_names)) - 1

        quality_pass = (
            dq_mask & required_dq_mask
        ) == required_dq_mask

        no_injection = (
            injection_mask & required_injection_mask
        ) == required_injection_mask

        passing_seconds = quality_pass & no_injection

        for second, passes in enumerate(passing_seconds):
            if passes:
                start = second * SAMPLE_RATE
                end = start + SAMPLE_RATE

                windows.append(strain[start:end])

                if gps_start is not None:
                    gps_times.append(gps_start + second)

    return windows, gps_times


def get_background_windows(file_paths):
    """Collect passing background windows from multiple files."""
    windows = []

    for file_path in file_paths:
        file_windows, _ = get_passing_windows(file_path)
        windows.extend(file_windows)

    return windows


def prepare_data(windows):
    """Convert strain windows into normalized PyTorch tensors."""
    data = torch.stack([
        torch.tensor(window, dtype=torch.float64)
        for window in windows
    ])

    data = data.unsqueeze(1)

    return normalize_windows(data)


def calculate_reconstruction_errors(model, data):
    """Calculate one reconstruction-error score per strain window."""
    model.eval()

    with torch.no_grad():
        reconstructed = model(data)

        errors = torch.mean(
            (data - reconstructed) ** 2,
            dim=(1, 2),
        )

    return errors


def calculate_threshold(model, file_paths):
    """Calculate a detector-specific background anomaly threshold."""
    windows = get_background_windows(file_paths)
    data = prepare_data(windows)

    errors = calculate_reconstruction_errors(
        model,
        data,
    )

    threshold = torch.quantile(
        errors,
        ANOMALY_PERCENTILE,
    ).item()

    return threshold, len(windows)


def get_conditioned_strain(
    file_path,
    start_sample,
    number_of_samples,
):
    """Load and condition a section of LIGO strain."""
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][
            start_sample:start_sample + number_of_samples
        ]

    strain = TimeSeries(
        strain,
        delta_t=1 / SAMPLE_RATE,
    )

    # Restrict the signal to a useful frequency band.
    strain = highpass(strain, 30.0)
    strain = lowpass(strain, 500.0)

    strain = torch.tensor(
        strain.numpy(),
        dtype=torch.float64,
    )

    # Standardize the conditioned strain.
    strain = (
        strain - strain.mean()
    ) / strain.std()

    return strain


def get_max_correlation(l1, h1):
    """Find the strongest H1-L1 correlation within +/-20 ms."""
    max_delay_samples = round(
        0.020 * SAMPLE_RATE
    )

    best_correlation = 0.0
    best_delay_samples = 0

    for delay in range(
        -max_delay_samples,
        max_delay_samples + 1,
    ):
        if delay < 0:
            l1_section = l1[-delay:]
            h1_section = h1[:delay]

        elif delay > 0:
            l1_section = l1[:-delay]
            h1_section = h1[delay:]

        else:
            l1_section = l1
            h1_section = h1

        correlation = torch.mean(
            l1_section * h1_section
        ).item()

        # Detector responses can have opposite signs,
        # so compare absolute correlation strength.
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_delay_samples = delay

    delay_ms = (
        best_delay_samples
        / SAMPLE_RATE
        * 1000
    )

    return best_correlation, delay_ms


def fine_time_comparison(
    l1_file_path,
    h1_file_path,
    candidate_gps,
    file_gps_start,
):
    """Condition and compare L1 and H1 strain near a candidate."""
    print("\nFine-time analysis:")
    print("Candidate GPS:", candidate_gps)

    # Analyze two seconds beginning at the candidate second.
    start_second = candidate_gps - file_gps_start

    start_sample = start_second * SAMPLE_RATE
    number_of_samples = SAMPLE_RATE * 2

    l1 = get_conditioned_strain(
        l1_file_path,
        start_sample,
        number_of_samples,
    )

    h1 = get_conditioned_strain(
        h1_file_path,
        start_sample,
        number_of_samples,
    )

    best_correlation, best_delay_ms = get_max_correlation(
        l1,
        h1,
    )

    print(
        "Best conditioned H1-L1 delay:",
        f"{best_delay_ms:.3f} ms",
    )

    print(
        "Conditioned correlation:",
        f"{best_correlation:.6f}",
    )

    # The maximum possible propagation delay between
    # Hanford and Livingston is approximately 10 ms.
    if abs(best_delay_ms) <= 10:
        print(
            "Delay is within the approximate "
            "physical H1-L1 range."
        )
    else:
        print(
            "Delay is outside the approximate "
            "physical H1-L1 range."
        )

    return best_delay_ms, best_correlation

def get_passing_seconds(file_path):
    """Return which seconds pass all quality and injection checks."""
    with h5py.File(file_path, "r") as file:
        dq_mask = file["quality/simple/DQmask"][:]
        dq_names = file["quality/simple/DQShortnames"][:]

        injection_mask = file["quality/injections/Injmask"][:]
        injection_names = file["quality/injections/InjShortnames"][:]

        required_dq_mask = (1 << len(dq_names)) - 1
        required_injection_mask = (
            1 << len(injection_names)
        ) - 1

        quality_pass = (
            dq_mask & required_dq_mask
        ) == required_dq_mask

        no_injection = (
            injection_mask & required_injection_mask
        ) == required_injection_mask

        return quality_pass & no_injection


def calculate_correlation_significance(
    candidate_correlation,
):
    """Compare candidate correlation with quality-passing background."""
    print("\nCorrelation significance test...")

    l1_background_file = (
        "data/L-L1_LOSC_4_V1-1126072320-4096.hdf5"
    )

    h1_background_file = (
        "data/H-H1_LOSC_4_V1-1126072320-4096.hdf5"
    )

    # Find seconds that are valid in BOTH detectors.
    l1_passing = get_passing_seconds(
        l1_background_file
    )

    h1_passing = get_passing_seconds(
        h1_background_file
    )

    common_length = min(
        len(l1_passing),
        len(h1_passing),
    )

    l1_passing = l1_passing[:common_length]
    h1_passing = h1_passing[:common_length]

    # We analyze two-second regions, so both consecutive
    # seconds must pass in both detectors.
    valid_start_seconds = []

    for second in range(common_length - 1):
        if (
            l1_passing[second]
            and l1_passing[second + 1]
            and h1_passing[second]
            and h1_passing[second + 1]
        ):
            valid_start_seconds.append(second)

    print(
        "Quality-passing paired regions:",
        len(valid_start_seconds),
    )

    if len(valid_start_seconds) == 0:
        print(
            "No shared quality-passing background "
            "regions available."
        )
        return None

    # Select comparisons evenly throughout the available
    # quality-passing background.
    comparison_count = min(
        BACKGROUND_COMPARISONS,
        len(valid_start_seconds),
    )

    if comparison_count == 1:
        selected_seconds = [
            valid_start_seconds[0]
        ]

    else:
        selected_seconds = []

        for index in range(comparison_count):
            position = round(
                index
                * (len(valid_start_seconds) - 1)
                / (comparison_count - 1)
            )

            selected_seconds.append(
                valid_start_seconds[position]
            )

    number_of_samples = SAMPLE_RATE * 2

    background_correlations = []

    for start_second in selected_seconds:
        start_sample = (
            start_second * SAMPLE_RATE
        )

        l1 = get_conditioned_strain(
            l1_background_file,
            start_sample,
            number_of_samples,
        )

        h1 = get_conditioned_strain(
            h1_background_file,
            start_sample,
            number_of_samples,
        )

        correlation, _ = get_max_correlation(
            l1,
            h1,
        )

        background_correlations.append(
            abs(correlation)
        )

    candidate_strength = abs(
        candidate_correlation
    )

    stronger_background = sum(
        correlation >= candidate_strength
        for correlation in background_correlations
    )

    empirical_p_value = (
        stronger_background + 1
    ) / (
        len(background_correlations) + 1
    )

    background_tensor = torch.tensor(
        background_correlations,
        dtype=torch.float64,
    )

    print(
        "Background comparisons:",
        len(background_correlations),
    )

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

    print(
        "Candidate correlation:",
        f"{candidate_strength:.6f}",
    )

    print(
        "Background regions >= candidate:",
        stronger_background,
    )

    print(
        "Empirical p-value:",
        f"{empirical_p_value:.6f}",
    )

    return empirical_p_value

def get_physical_strain(
    file_path,
    start_sample,
    number_of_samples,
):
    """Load calibrated detector strain without standardizing it."""
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][
            start_sample:start_sample + number_of_samples
        ]

    return torch.tensor(
        strain,
        dtype=torch.float64,
    )

def get_conditioned_physical_strain(
    file_path,
    start_sample,
    number_of_samples,
):
    """Band-limit calibrated strain while preserving its physical scale."""
    strain = get_physical_strain(
        file_path,
        start_sample,
        number_of_samples,
    )

    strain = TimeSeries(
        strain.numpy(),
        delta_t=1 / SAMPLE_RATE,
    )

    strain = highpass(strain, 30.0)
    strain = lowpass(strain, 500.0)

    return torch.tensor(
        strain.numpy(),
        dtype=torch.float64,
    )

def characterize_candidate(
    file_path,
    candidate_gps,
    file_gps_start,
):
    """Measure basic properties of a surviving strain candidate."""
    start_second = candidate_gps - file_gps_start
    start_sample = start_second * SAMPLE_RATE

    physical_strain = get_conditioned_physical_strain(
        file_path,
        start_sample,
        SAMPLE_RATE * 2,
    )

    peak_physical_strain = torch.max(
        torch.abs(physical_strain)
    ).item()

    strain = get_conditioned_strain(
        file_path,
        start_sample,
        SAMPLE_RATE * 2,
    )

    peak_amplitude = torch.max(
        torch.abs(strain)
    ).item()

    frequency_spectrum = torch.fft.rfft(strain)

    frequencies = torch.fft.rfftfreq(
        len(strain),
        d=1 / SAMPLE_RATE,
    )

    power = torch.abs(
        frequency_spectrum
    ) ** 2

    # Ignore the zero-frequency component.
    dominant_index = torch.argmax(
        power[1:]
    ).item() + 1

    dominant_frequency = frequencies[
        dominant_index
    ].item()

    print("\nSignal characterization:")
    print("GPS:", candidate_gps)
    print(
        "Peak conditioned amplitude:",
        f"{peak_amplitude:.6f}",
    )

    print(
        "Peak physical strain:",
        f"{peak_physical_strain:.6e}",
    )

    print(
        "Dominant frequency:",
        f"{dominant_frequency:.2f} Hz",
    )

    return {
        "peak_conditioned_amplitude": peak_amplitude,
        "peak_physical_strain": peak_physical_strain,
        "dominant_frequency_hz": dominant_frequency,
    }

def main():
    """Run the complete cross-detector anomaly-detection pipeline."""
    print("Loading trained autoencoder...")

    model = StrainAutoencoder()

    model.load_state_dict(
        torch.load(
            "strain_autoencoder.pth",
            map_location="cpu",
        )
    )

    model.eval()

    # ---------------------------------------------------------
    # Detector-specific anomaly thresholds
    # ---------------------------------------------------------

    print("Calculating L1 background threshold...")

    l1_threshold, l1_background_count = calculate_threshold(
        model,
        L1_BACKGROUND_FILES,
    )

    print(
        f"L1 threshold: {l1_threshold:.6f} "
        f"from {l1_background_count} background windows"
    )

    print("Calculating H1 background threshold...")

    h1_threshold, h1_background_count = calculate_threshold(
        model,
        H1_BACKGROUND_FILES,
    )

    print(
        f"H1 threshold: {h1_threshold:.6f} "
        f"from {h1_background_count} background windows"
    )

    # ---------------------------------------------------------
    # L1 anomaly scan
    # ---------------------------------------------------------

    print("\nScanning unseen L1 data...")

    l1_windows, l1_gps_times = get_passing_windows(
        L1_TEST_FILE,
        TEST_GPS_START,
    )

    l1_data = prepare_data(l1_windows)

    l1_errors = calculate_reconstruction_errors(
        model,
        l1_data,
    )

    l1_candidates = {}

    for index, gps_time in enumerate(l1_gps_times):
        error = l1_errors[index].item()

        if error > l1_threshold:
            l1_candidates[gps_time] = error

    print(
        "L1 candidate anomalies:",
        len(l1_candidates),
    )

    for gps_time, error in l1_candidates.items():
        print(
            "GPS:",
            gps_time,
            "| L1 reconstruction error:",
            f"{error:.6f}",
        )

    # ---------------------------------------------------------
    # H1 scan
    # ---------------------------------------------------------

    print("\nScanning unseen H1 data...")

    h1_windows, h1_gps_times = get_passing_windows(
        H1_TEST_FILE,
        TEST_GPS_START,
    )

    h1_data = prepare_data(h1_windows)

    h1_errors = calculate_reconstruction_errors(
        model,
        h1_data,
    )

    h1_error_by_gps = {
        gps_time: h1_errors[index].item()
        for index, gps_time in enumerate(h1_gps_times)
    }

    # ---------------------------------------------------------
    # Cross-detector anomaly check
    # ---------------------------------------------------------

    print("\nCross-detector candidate check:")

    confirmed_candidates = []

    for gps_time, l1_error in l1_candidates.items():
        print(f"\nL1 candidate GPS {gps_time}")
        print(f"L1 error: {l1_error:.6f}")

        h1_matches = []

        # The one-second autoencoder windows are coarse,
        # so inspect the same and adjacent GPS seconds.
        for check_time in range(
            gps_time - 1,
            gps_time + 2,
        ):
            if check_time in h1_error_by_gps:
                h1_error = h1_error_by_gps[
                    check_time
                ]

                if h1_error > h1_threshold:
                    h1_matches.append(
                        (check_time, h1_error)
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

            confirmed_candidates.append(
                gps_time
            )

        else:
            print(
                "No H1 anomaly found nearby."
            )

    print()

    print(
        "Cross-detector candidates:",
        len(confirmed_candidates),
    )

    for gps_time in confirmed_candidates:
        print("GPS:", gps_time)

    # ---------------------------------------------------------
    # Fine-time and statistical validation
    # ---------------------------------------------------------

    surviving_candidates = []

    for gps_time in confirmed_candidates:
        delay_ms, correlation = fine_time_comparison(
            L1_TEST_FILE,
            H1_TEST_FILE,
            gps_time,
            TEST_GPS_START,
        )

        p_value = calculate_correlation_significance(
            correlation,
        )

        print("\nCandidate classification:")
        print("GPS:", gps_time)

        # Reject candidates whose strongest correlation occurs
        # outside the physically possible H1-L1 delay.
        if abs(delay_ms) > MAX_PHYSICAL_DELAY_MS:
            print("REJECTED")
            print(
                "Reason: H1-L1 delay is outside "
                "the physical range."
            )
            continue

        if p_value is None:
            print("REJECTED")
            print(
                "Reason: insufficient valid background "
                "for significance testing."
            )
            continue

        # Project-level screening criterion.
        # This is not formal LIGO detection significance.
        if p_value >= SIGNIFICANCE_THRESHOLD:
            print("REJECTED")
            print(
                "Reason: cross-detector correlation is "
                "compatible with ordinary background."
            )
            continue

        print("PASSED")
        print(
            "Candidate survived cross-detector "
            "validation."
        )

        characterization = characterize_candidate(
            L1_TEST_FILE,
            gps_time,
            TEST_GPS_START,
        )

        physics_result = analyze_candidate_physics(
            characterization["peak_physical_strain"]
        )

        surviving_candidates.append(
            {
                "gps": gps_time,
                "delay_ms": delay_ms,
                "correlation": correlation,
                "p_value": p_value,
                "characterization": characterization,
                "physics": physics_result,
            }
        )

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    print("\n========================================")
    print("FINAL DETECTION RESULTS")
    print("========================================")

    print(
        "Initial L1 anomalies:",
        len(l1_candidates),
    )

    print(
        "Cross-detector candidates:",
        len(confirmed_candidates),
    )

    print(
        "Surviving candidates:",
        len(surviving_candidates),
    )

    if surviving_candidates:
        print(
            "\nCandidates ready for "
            "signal characterization:"
        )

        for candidate in surviving_candidates:
            print()
            print(
                "GPS:",
                candidate["gps"],
            )
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
        print(
            "\nNo candidates survived validation."
        )


if __name__ == "__main__":
    main()