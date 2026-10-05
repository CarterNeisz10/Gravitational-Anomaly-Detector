"""
Generate simulated Advanced LIGO background strain for autoencoder training.

This module uses PyCBC to generate noise based on the Advanced LIGO design
sensitivity power spectral density (PSD). The simulated strain is used as the
first stage of training for the convolutional autoencoder.
"""

from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower


# Simulation settings
SAMPLE_RATE = 4096
DURATION = 4096

DELTA_T = 1 / SAMPLE_RATE
NUM_SAMPLES = SAMPLE_RATE * DURATION
DELTA_F = 1 / DURATION

LOW_FREQUENCY_CUTOFF = 20

"""
    Generate simulated Advanced LIGO background strain.

    Creates a power spectral density based on the Advanced LIGO design
    sensitivity curve and uses it to generate reproducible Gaussian noise.

    Returns:
        TimeSeries: Simulated strain sampled at 4096 Hz.
    """
def generate_background_noise():
    psd = aLIGOZeroDetHighPower(
        NUM_SAMPLES // 2 + 1,
        DELTA_F,
        LOW_FREQUENCY_CUTOFF,
    )

    return noise_from_psd(
        NUM_SAMPLES,
        DELTA_T,
        psd,
        seed=42,
    )

"""
    Split strain data into consecutive, non-overlapping windows.

    By default, each window contains 4096 measurements, corresponding to
    one second of strain data at the project's 4096 Hz sample rate.

    Args:
        strain: Sequence of strain measurements.
        window_size: Number of measurements in each window.

    Returns:
        list: Complete strain windows. Any incomplete final window is ignored.
    """
def create_windows(strain, window_size=SAMPLE_RATE):
    return [
        strain[start:start + window_size]
        for start in range(
            0,
            len(strain) - window_size + 1,
            window_size,
        )
    ]