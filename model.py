"""
Convolutional autoencoder for detecting anomalies in LIGO strain data.

The model is trained in two stages. It first learns to reconstruct simulated
Advanced LIGO background noise, then continues training on quality-filtered
real LIGO strain. Poor reconstruction of unseen strain can later be used as
an indication that a window differs from the learned background.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from simulation import generate_background_noise, create_windows
from Main import get_real_training_windows


# Training settings
BATCH_SIZE = 32
SIMULATED_EPOCHS = 10
REAL_EPOCHS = 5
LEARNING_RATE = 0.001


"""
    Normalize each strain window independently.

    Each window is transformed to have approximately zero mean and unit
    standard deviation. This prevents differences in absolute scale from
    dominating autoencoder training.

    Args:
        windows: PyTorch tensor containing strain windows.

    Returns:
        Tensor: Normalized strain windows as 32-bit floating-point values.
"""
def normalize_windows(windows):
    windows = windows.double()

    mean = windows.mean(dim=2, keepdim=True)
    std = windows.std(dim=2, keepdim=True)

    return ((windows - mean) / std).float()


"""
    One-dimensional convolutional autoencoder for LIGO strain data.

    The encoder compresses each 4096-sample strain window through three
    convolutional layers. The decoder mirrors this structure using
    transposed convolutions to reconstruct the original strain window.
"""
class StrainAutoencoder(nn.Module):

    """
        Initialize the encoder and decoder layers.
    """
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
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=8, stride=4, padding=2),
        )

    """
        Reconstruct a strain window using the autoencoder.

        Args:
            strain: Batch of normalized strain windows.

        Returns:
            Tensor: Reconstructed strain windows.
    """
    def forward(self, strain):
        encoded = self.encoder(strain)
        return self.decoder(encoded)


"""
    Convert strain windows into normalized PyTorch tensors.

    The individual windows are stacked into a batch, given a single input
    channel for the Conv1d layers, and normalized independently.

    Args:
        windows: Collection of strain windows.

    Returns:
        Tensor: Normalized strain data with shape
        (number of windows, 1, samples per window).
"""
def prepare_windows(windows):
    data = torch.stack([
        torch.tensor(window, dtype=torch.float64)
        for window in windows
    ])

    return normalize_windows(data.unsqueeze(1))


"""
    Train the autoencoder to reconstruct background strain.

    Training minimizes mean squared reconstruction error using the Adam
    optimizer. The same function is used for both simulated-background
    training and real-background fine-tuning.

    Args:
        model: StrainAutoencoder being trained.
        training_data: Normalized strain windows used for training.
        epochs: Number of complete passes through the training data.
        stage_name: Description printed before the training stage begins.
"""
def train_model(model, training_data, epochs, stage_name):
    dataset = TensorDataset(training_data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # Stage 1: train on simulated Advanced LIGO background noise.
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

    # Stage 2: continue training on quality-filtered real LIGO background.
    print("\nLoading real LIGO background...")

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

    # Save only the trained model parameters rather than the full model object.
    torch.save(model.state_dict(), "strain_autoencoder.pth")

    print("\nTrained model saved to strain_autoencoder.pth")