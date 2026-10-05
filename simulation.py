"""Generate simulated LIGO background strain for autoencoder training."""

from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower


SAMPLE_RATE = 4096
DURATION = 4096

DELTA_T = 1 / SAMPLE_RATE
NUM_SAMPLES = SAMPLE_RATE * DURATION
DELTA_F = 1 / DURATION

LOW_FREQUENCY_CUTOFF = 20


def generate_background_noise():
    """Generate simulated Advanced LIGO background strain."""

    psd = aLIGOZeroDetHighPower(
        NUM_SAMPLES // 2 + 1,
        DELTA_F,
        LOW_FREQUENCY_CUTOFF,
    )

    noise = noise_from_psd(
        NUM_SAMPLES,
        DELTA_T,
        psd,
        seed=42,
    )

    return noise


def create_windows(strain, window_size=SAMPLE_RATE):
    """Split strain measurements into fixed one-second windows."""
    windows = []

    for start in range(0, len(strain) - window_size + 1, window_size):
        end = start + window_size
        windows.append(strain[start:end])

    return windows


if __name__ == "__main__":
    background = generate_background_noise()
    windows = create_windows(background)

    print("Simulated measurements:", len(background))
    print("Duration:", len(background) / SAMPLE_RATE, "seconds")
    print("Training windows:", len(windows))
    print("Measurements per window:", len(windows[0]))