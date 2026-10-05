"""
Locate, filter, and prepare real LIGO strain data from GWOSC.

This module queries the Gravitational Wave Open Science Center (GWOSC) for
LIGO strain files and loads locally downloaded HDF5 data. Data-quality and
hardware-injection flags are used to retain only suitable background strain
for the real-data stage of autoencoder training.
"""

import json
import subprocess

import h5py

from simulation import create_windows


# GWOSC search settings
GWOSC_API = "https://gwosc.org/api/v2/strain-files"

DETECTOR = "L1"
START_TIME = 1126068224
END_TIME = 1126072320
SAMPLE_RATE_KHZ = 4
SAMPLES_PER_SECOND = SAMPLE_RATE_KHZ * 1024

# Local L1 files used for real-background training
DATA_FILES = [
    "data/L-L1_LOSC_4_V1-1126068224-4096.hdf5",
    "data/L-L1_LOSC_4_V1-1126072320-4096.hdf5",
]


"""
    Query GWOSC for strain files covering the configured detector and time range.

    Returns:
        dict: Decoded JSON response containing matching GWOSC strain files.
"""
def get_strain_files():
    url = (
        f"{GWOSC_API}?start={START_TIME}"
        f"&stop={END_TIME}"
        f"&detector={DETECTOR}"
        f"&sample-rate={SAMPLE_RATE_KHZ}"
        "&pagesize=10"
    )

    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


"""
    Create a bitmask that requires every supplied quality flag.

    Each flag occupies one bit in the GWOSC quality mask. Setting every
    corresponding bit creates a mask that can be used to require all listed
    data-quality or no-hardware-injection conditions simultaneously.

    Args:
        flag_names: Collection of flag names represented by the mask.

    Returns:
        int: Bitmask with one required bit for every supplied flag.
"""
def get_required_mask(flag_names):
    return (1 << len(flag_names)) - 1


"""
    Load strain measurements from seconds that pass all quality checks.

    A second is retained only when every available data-quality flag passes
    and every no-hardware-injection flag is set. Passing seconds are converted
    from quality-mask indices into their corresponding strain samples.

    Args:
        file_path: Path to a local GWOSC HDF5 strain file.

    Returns:
        list: Strain measurements from all passing seconds.
"""
def get_passing_strain(file_path):
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][:]

        dq_mask = file["quality/simple/DQmask"][:]
        dq_names = file["quality/simple/DQShortnames"][:]

        injection_mask = file["quality/injections/Injmask"][:]
        injection_names = file["quality/injections/InjShortnames"][:]

        required_dq_mask = get_required_mask(dq_names)
        required_injection_mask = get_required_mask(injection_names)

        quality_pass = (dq_mask & required_dq_mask) == required_dq_mask
        no_injection = (
            injection_mask & required_injection_mask
        ) == required_injection_mask

        passing_seconds = quality_pass & no_injection
        passing_strain = []

        for second, passes in enumerate(passing_seconds):
            if passes:
                start = second * SAMPLES_PER_SECOND
                end = start + SAMPLES_PER_SECOND
                passing_strain.extend(strain[start:end])

    return passing_strain


"""
    Prepare quality-filtered real LIGO background for autoencoder training.

    Strain passing the quality requirements is collected from each configured
    L1 training file and divided into consecutive one-second windows.

    Returns:
        list: Quality-filtered one-second LIGO strain windows.
"""
def get_real_training_windows():
    passing_strain = []

    for file_path in DATA_FILES:
        passing_strain.extend(get_passing_strain(file_path))

    return create_windows(passing_strain)


"""
    Display available GWOSC files and summarize the prepared training data.

    This provides a standalone check of the GWOSC query and reports the amount
    of quality-filtered real strain available for autoencoder training.
"""
def main():
    print(f"Searching GWOSC for {DETECTOR} strain data...")

    response = get_strain_files()
    strain_files = response["results"]

    print(f"Found {response['results_count']} strain file(s).\n")

    for strain_file in strain_files:
        print("Detector:", strain_file["detector"])
        print("GPS start:", strain_file["gps_start"])
        print("Sample rate:", strain_file["sample_rate_kHz"], "kHz")
        print("HDF5:", strain_file["hdf5_url"])
        print()

    real_windows = get_real_training_windows()

    print(
        "Passing strain measurements:",
        len(real_windows) * SAMPLES_PER_SECOND,
    )
    print("Passing strain duration:", len(real_windows), "seconds")
    print("Real training windows:", len(real_windows))
    print("Measurements per window:", len(real_windows[0]))


if __name__ == "__main__":
    main()