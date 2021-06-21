import argparse
from pathlib import Path

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from train import train


def main(num_epochs=50, root_path=Path("."), batch_size=100):
    figures_path = root_path / "figures"
    figures_path.mkdir(exist_ok=True)
    data_path = root_path / "data"
    data_path.mkdir(exist_ok=True)
    models_path = root_path / "models"
    models_path.mkdir(exist_ok=True)

    # MNIST dataset
    dataset = dset.MNIST(data_path, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        dataloader=dataloader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        models_path=models_path,
        figures_path=figures_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InfoGAN training")
    parser.add_argument(
        "--root-path",
        type=str,
        help="Path for putting data, figures and saved models",
        required=True,
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    main(
        num_epochs=args.num_epochs,
        root_path=Path(args.root_path),
        batch_size=args.batch_size,
    )
