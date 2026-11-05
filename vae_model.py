"""Variational Autoencoder (VAE) for review anomaly detection.

Module 5 – Generative Models
Architecture: Encoder (µ, logσ²) → Reparameterization → Decoder → Reconstruction
Anomaly score = per-sample reconstruction error (MSE).
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Maps a TF-IDF / embedding vector to latent Gaussian parameters (µ, logσ²)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Reconstructs the original input vector from a latent sample z."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),   # input is normalised to [0,1] via MinMaxScaler
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReviewVAE(nn.Module):
    """Full VAE wrapping Encoder + reparameterisation trick + Decoder.

    Args:
        input_dim:  Dimensionality of the TF-IDF feature vector.
        hidden_dim: Width of the intermediate hidden layers.
        latent_dim: Dimensionality of the latent space z.
        beta:       Weight on the KL-divergence term (beta-VAE formulation).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = µ + ε·σ  where  ε ~ N(0,I)  (only during training)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference time

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, mu, logvar, z)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO = Reconstruction loss + beta * KL divergence.

        Returns (total_loss, reconstruction_loss, kl_loss).
        """
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        # Closed-form KL between N(µ,σ²) and N(0,1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction (anomaly score)."""
        self.eval()
        recon, _, _, _ = self(x)
        return nn.functional.mse_loss(recon, x, reduction="none").mean(dim=1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean µ for visualisation or downstream tasks."""
        self.eval()
        mu, _ = self.encoder(x)
        return mu
