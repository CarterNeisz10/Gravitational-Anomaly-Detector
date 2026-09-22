"""Locate and filter LIGO strain data available through GWOSC."""

import json
import subprocess

import h5py


GWOSC_API = "https://gwosc.org/api/v2/strain-files"

DETECTOR = "L1"
START_TIME = 1126068224
END_TIME = 1126072320
SAMPLE_RATE_KHZ = 4
SAMPLES_PER_SECOND = SAMPLE_RATE_KHZ * 1024

DATA_FILE = "data/L-L1_LOSC_4_V1-1126068224-4096.hdf5"


def get_strain_files():
    """
    Query GWOSC for strain files covering the configured detector and time range.

    Returns:
        dict: Parsed GWOSC API response containing matching strain-file metadata.

    Raises:
        subprocess.CalledProcessError: If the GWOSC request fails.
        json.JSONDecodeError: If GWOSC returns an invalid JSON response.
    """
    url = (
        f"{GWOSC_API}"
        f"?start={START_TIME}"
        f"&stop={END_TIME}"
        f"&detector={DETECTOR}"
        f"&sample-rate={SAMPLE_RATE_KHZ}"
        f"&pagesize=10"
    )

    # curl is used because direct Python HTTPS requests to GWOSC are
    # unreliable in the current local development environment.
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def get_required_mask(flag_names):
    """Create a bitmask requiring every listed flag to pass."""
    required_mask = 0

    for index in range(len(flag_names)):
        required_mask |= 1 << index

    return required_mask


def get_passing_strain(file_path):
    """Return strain measurements from seconds that pass all quality checks."""
    with h5py.File(file_path, "r") as file:
        strain = file["strain/Strain"][:]

        dq_mask = file["quality/simple/DQmask"][:]
        dq_names = file["quality/simple/DQShortnames"][:]

        injection_mask = file["quality/injections/Injmask"][:]
        injection_names = file["quality/injections/InjShortnames"][:]

        required_dq_mask = get_required_mask(dq_names)
        required_injection_mask = get_required_mask(injection_names)

        # A second passes only when every required quality bit is set.
        quality_pass = (dq_mask & required_dq_mask) == required_dq_mask

        # These flags being set means no hardware injection is present.
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


def main():
    """Locate GWOSC files and extract strain from passing data."""
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

    passing_strain = get_passing_strain(DATA_FILE)

    print("Passing strain measurements:", len(passing_strain))


if __name__ == "__main__":
    main()