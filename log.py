"""Log training progress"""
from pathlib import Path

import torch

from plot import plot_categorical_vs_continuous


def log(
    generator: torch.nn.Module,
    epoch: int,
    iteration: int,
    discriminator_loss: float,
    generator_loss: float,
    device: torch.device,
    models_path: Path,
    figures_path: Path,
):
    print(
        f"Epoch: {epoch}, "
        f"Discriminator loss: {discriminator_loss}, "
        f"Generator loss: {generator_loss}"
    )
    prefix = f"{epoch:03}_"
    generator.eval()
    plot_categorical_vs_continuous(
        generator=generator,
        figures_path=figures_path,
        prefix=prefix,
        device=device,
    )
    generator.train()
    torch.save(
        generator.state_dict(),
        models_path / f"{prefix}generator.pt",
    )
