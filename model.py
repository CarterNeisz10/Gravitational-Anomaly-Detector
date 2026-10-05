"""Autoencoder model for detecting anomalies in LIGO strain data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from simulation import generate_background_noise, create_windows
from Main import get_real_training_windows


BATCH_SIZE = 32
SIMULATED_EPOCHS = 10
REAL_EPOCHS = 5
LEARNING_RATE = 0.001


def normalize_windows(windows):
    """Normalize each strain window to zero mean and unit standard deviation."""
    windows = windows.double()

    mean = windows.mean(dim=2, keepdim=True)
    std = windows.std(dim=2, keepdim=True)

    return ((windows - mean) / std).float()


class StrainAutoencoder(nn.Module):
    """One-dimensional convolutional autoencoder for LIGO strain."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32, kernel_size=8, stride=4, padding=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 16, kernel_size=8, stride=4, padding=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16, 1, kernel_size=8, stride=4, padding=2
            ),
        )

    def forward(self, strain):
        """Encode and reconstruct a strain window."""
        encoded = self.encoder(strain)
        reconstructed = self.decoder(encoded)

        return reconstructed


def prepare_windows(windows):
    """Convert strain windows into normalized PyTorch tensors."""
    data = torch.stack([
        torch.tensor(window, dtype=torch.float64)
        for window in windows
    ])

    data = data.unsqueeze(1)

    return normalize_windows(data)


def train_model(model, training_data, epochs, stage_name):
    """Train the autoencoder to reconstruct background strain."""
    dataset = TensorDataset(training_data)

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    print(stage_name)

    for epoch in range(epochs):
        total_loss = 0

        for (batch,) in data_loader:
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- reconstruction loss: {average_loss:.6f}"
        )


if __name__ == "__main__":
    model = StrainAutoencoder()

    # Stage 1: simulated LIGO background
    print("Generating simulated LIGO background...")

    simulated_background = generate_background_noise()
    simulated_windows = create_windows(simulated_background)
    simulated_data = prepare_windows(simulated_windows)

    print("Simulated data shape:", simulated_data.shape)
    print()

    train_model(
        model,
        simulated_data,
        SIMULATED_EPOCHS,
        "Stage 1: simulated background training",
    )

    print()

    # Stage 2: real quality-filtered LIGO background
    print("Loading real LIGO background...")

    real_windows = get_real_training_windows()
    real_data = prepare_windows(real_windows)

    print("Real data shape:", real_data.shape)
    print()

    train_model(
        model,
        real_data,
        REAL_EPOCHS,
        "Stage 2: real LIGO background training",
    )

    torch.save(model.state_dict(), "strain_autoencoder.pth")

    print()
    print("Trained model saved to strain_autoencoder.pth")