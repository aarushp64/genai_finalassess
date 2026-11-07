"""
NeuroTutor AI - GAN Educational Explorer & PyTorch Demonstration
Author: Aarush Parate (Roll No: 1)
Unit II: GANs and Large Language Models
Interactive PyTorch GAN training on 1D/2D continuous manifolds with real vs generated sample tracking and loss curves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple


class Generator(nn.Module):
    """Generates synthetic samples from latent Gaussian noise z ~ N(0, 1)."""

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 32, output_dim: int = 1):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Classifies samples as real (1.0) vs generated fake (0.0)."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANExplorerEngine:
    """Manages PyTorch GAN model instantiation, training steps, and visualization data."""

    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        self.reset_models()

    def reset_models(self):
        """Resets network weights and training histories."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.generator = Generator(latent_dim=self.latent_dim, hidden_dim=32, output_dim=1)
        self.discriminator = Discriminator(input_dim=1, hidden_dim=32)
        self.criterion = nn.BCELoss()
        self.opt_g = optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.epoch_history = []
        self.d_loss_history = []
        self.g_loss_history = []
        self.total_epochs = 0

    @staticmethod
    def sample_real_data(batch_size: int = 128) -> torch.Tensor:
        """
        Target Ground Truth Distribution: Bimodal Gaussian Mixture.
        p(x) = 0.5 * N(-2.5, 0.5^2) + 0.5 * N(2.5, 0.5^2)
        """
        coin = np.random.binomial(1, 0.5, batch_size)
        mode1 = np.random.normal(-2.5, 0.5, batch_size)
        mode2 = np.random.normal(2.5, 0.5, batch_size)
        samples = coin * mode1 + (1 - coin) * mode2
        return torch.tensor(samples, dtype=torch.float32).unsqueeze(1)

    def train_steps(self, num_epochs: int = 25, batch_size: int = 128) -> Dict[str, Any]:
        """
        Executes adversarial training loop over specified number of iterations.
        Returns training metrics and sample distributions.
        """
        for _ in range(num_epochs):
            self.total_epochs += 1

            # ---------------------
            # Train Discriminator
            # ---------------------
            self.opt_d.zero_grad()
            real_data = self.sample_real_data(batch_size)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Discriminator on real
            out_real = self.discriminator(real_data)
            loss_d_real = self.criterion(out_real, real_labels)

            # Discriminator on fake
            z = torch.randn(batch_size, self.latent_dim)
            fake_data = self.generator(z)
            out_fake = self.discriminator(fake_data.detach())
            loss_d_fake = self.criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.opt_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            self.opt_g.zero_grad()
            # Generator wants discriminator to predict 1.0 for fake samples
            out_g = self.discriminator(fake_data)
            loss_g = self.criterion(out_g, real_labels)
            loss_g.backward()
            self.opt_g.step()

            # Record history
            self.epoch_history.append(self.total_epochs)
            self.d_loss_history.append(float(loss_d.item()))
            self.g_loss_history.append(float(loss_g.item()))

        return {
            "current_epoch": self.total_epochs,
            "d_loss": self.d_loss_history[-1],
            "g_loss": self.g_loss_history[-1],
        }

    def generate_distribution_plot(self, num_samples: int = 400) -> go.Figure:
        """Visualizes real bimodal distribution vs GAN generated samples."""
        self.generator.eval()
        with torch.no_grad():
            real_samples = self.sample_real_data(num_samples).squeeze().numpy()
            z = torch.randn(num_samples, self.latent_dim)
            generated_samples = self.generator(z).squeeze().numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=real_samples,
                name="Real Target Distribution (Bimodal Gaussian)",
                opacity=0.65,
                marker_color="#8BAFD0",
                nbinsx=40,
                histnorm="probability density",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=generated_samples,
                name="GAN Generated Samples G(z)",
                opacity=0.65,
                marker_color="#FFA726",
                nbinsx=40,
                histnorm="probability density",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Real vs GAN Generated Distributions (Trained {self.total_epochs} Epochs)",
                font=dict(color="#F2F4F7", size=14),
            ),
            barmode="overlay",
            xaxis=dict(title="Value (x)", color="#AAB5C4", gridcolor="#39465A"),
            yaxis=dict(title="Probability Density", color="#AAB5C4", gridcolor="#39465A"),
            paper_bgcolor="#171D29",
            plot_bgcolor="#202838",
            legend=dict(font=dict(color="#F2F4F7"), orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=40, t=60, b=40),
            height=380,
        )
        return fig

    def generate_loss_curve(self) -> go.Figure:
        """Visualizes adversarial min-max loss convergence."""
        fig = go.Figure()
        if self.epoch_history:
            fig.add_trace(
                go.Scatter(
                    x=self.epoch_history,
                    y=self.d_loss_history,
                    mode="lines",
                    name="Discriminator Loss (L_D)",
                    line=dict(color="#EF5350", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.epoch_history,
                    y=self.g_loss_history,
                    mode="lines",
                    name="Generator Loss (L_G)",
                    line=dict(color="#81C784", width=2),
                )
            )

        fig.update_layout(
            title=dict(text="Adversarial Loss History", font=dict(color="#F2F4F7", size=14)),
            xaxis=dict(title="Training Steps / Epochs", color="#AAB5C4", gridcolor="#39465A"),
            yaxis=dict(title="Binary Cross-Entropy Loss", color="#AAB5C4", gridcolor="#39465A"),
            paper_bgcolor="#171D29",
            plot_bgcolor="#202838",
            legend=dict(font=dict(color="#F2F4F7")),
            margin=dict(l=40, r=40, t=50, b=40),
            height=300,
        )
        return fig
