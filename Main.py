"""Locate LIGO strain data available through the GWOSC public API."""

import json
import subprocess


GWOSC_API = "https://gwosc.org/api/v2/strain-files"

DETECTOR = "L1"
START_TIME = 1126068224
END_TIME = 1126072320
SAMPLE_RATE_KHZ = 4


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


def main():
    """Locate and display GWOSC strain files matching the configured query."""
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


if __name__ == "__main__":
    main()