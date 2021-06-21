import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

from model import Generator


def load_generator(models_path: Path):
    """load the trained generator"""
    generator = Generator()
    generator.load_state_dict(
        torch.load(sorted(models_path.glob("*_generator.pt"))[-1])
    )
    generator.eval()
    return generator


def plot_digit(generator, z, cont0, cont1, digit, ax, device):
    """Generate an image and plot it to a given axis"""
    one_hot = np.zeros((10, 1))
    one_hot = torch.zeros((1, 10)).to(device)
    one_hot[:, digit] = 1
    c = torch.empty(1, 2).to(device)
    c[:, 0] = cont0
    c[:, 1] = cont1
    img = generator(torch.cat([z, one_hot, c], dim=1)).data.cpu().view(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_categorical_vs_continuous(
    generator: torch.nn.Module,
    figures_path: Path,
    device: torch.device,
    prefix: str = "",
):
    """
    Plot two grids, one for each continuous latent code.
    x axis is the value of one of the continuous latent codes.
    y axis is the categorical latent code.
    """
    z = torch.Tensor(1, 62).uniform_(-1, 1).to(device)
    ncols = 11
    for c_index in range(2):
        fig = plt.figure(figsize=(10, ncols - 1))
        spec = gridspec.GridSpec(ncols=ncols, nrows=10, figure=fig)
        for digit in range(10):
            for c_step, c in enumerate(np.linspace(-1, 1, ncols)):
                ax = fig.add_subplot(spec[digit, c_step])
                if digit == 9:
                    ax.set_xlabel(f"{c:.2}")
                plot_digit(
                    generator=generator.to(device),
                    z=z,
                    cont0=0 if c_index else c,
                    cont1=0 if not c_index else c,
                    digit=digit,
                    ax=ax,
                    device=device,
                )
        plt.savefig(figures_path / f"{prefix}categorical_vs_continuous{c_index}.png")
        plt.close()


def plot_continuous_grid(generator, figures_path, device):
    """
    Plot ten grids, one for each digit.
    Each axis is the value of one of the continuous latent codes.
    """
    z = torch.Tensor(1, 62).uniform_(-1, 1).to(device)
    ncols = 10
    for digit in range(10):
        fig = plt.figure(figsize=(ncols, ncols))
        spec = gridspec.GridSpec(ncols=ncols, nrows=10, figure=fig)
        for c0_step, cont0 in enumerate(np.linspace(-1, 1, ncols)):
            for c1_step, cont1 in enumerate(np.linspace(-1, 1, ncols)):
                ax = fig.add_subplot(spec[c0_step, c1_step])
                plot_digit(
                    generator=generator.to(device),
                    z=z,
                    cont0=cont0,
                    cont1=cont1,
                    digit=digit,
                    ax=ax,
                    device=device,
                )
        plt.savefig(figures_path / f"continuous_grid_{digit}.png")
        plt.close()


def main(root_path):
    figures_path = root_path / "figures"
    generator = load_generator(models_path=root_path / "models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_categorical_vs_continuous(
        generator=generator,
        figures_path=figures_path,
        device=device,
    )
    plot_continuous_grid(
        generator=generator,
        figures_path=figures_path,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot some figures")
    parser.add_argument(
        "--root-path",
        type=str,
        help="Path for putting data, figures and saved models",
        required=True,
    )
    args = parser.parse_args()
    main(Path(args.root_path))
