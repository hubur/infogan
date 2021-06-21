from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from log import log
from model import Discriminator, Generator, RecognitionNetwork, SharedLayers


def train(
    dataloader: DataLoader,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    models_path: Path,
    figures_path: Path,
):
    """Run InfoGAN training"""

    # Initialize models
    shared_layers = SharedLayers().to(device)
    discriminator = Discriminator().to(device)
    recognition_network = RecognitionNetwork().to(device)
    generator = Generator().to(device)

    # Define loss functions
    criterion_d = nn.BCELoss().to(device)
    criterion_q_discrete = nn.CrossEntropyLoss().to(device)
    criterion_q_continous = nn.GaussianNLLLoss().to(device)

    # Adam optimizer
    optimizer_d = optim.Adam(
        [
            {"params": shared_layers.parameters()},
            {"params": discriminator.parameters()},
        ],
        lr=0.0002,
        betas=(0.5, 0.99),
    )
    optimizer_g_q = optim.Adam(
        [
            {"params": generator.parameters()},
            {"params": recognition_network.parameters()},
        ],
        lr=0.001,
        betas=(0.5, 0.99),
    )

    # Training loop
    for epoch in range(num_epochs):
        for iteration, data in enumerate(dataloader, 0):
            # sample noise and latent code
            noise = torch.empty(batch_size, 62, device=device).uniform_(-1, 1)
            digits = np.random.randint(10, size=batch_size)
            discrete_latent_code = np.zeros((batch_size, 10), dtype=np.float32)
            discrete_latent_code[range(batch_size), digits] = 1.0
            discrete_latent_code = torch.tensor(discrete_latent_code).to(device)
            continuous_latent_code = torch.empty(batch_size, 2, device=device).uniform_(
                -1, 1
            )

            # generate images
            generated_images = generator(
                torch.cat(
                    [noise, discrete_latent_code, continuous_latent_code],
                    axis=1,
                )
            )

            # train step Discriminator
            optimizer_d.zero_grad()
            loss_real = criterion_d(
                discriminator(shared_layers(data[0].to(device))),
                torch.ones(batch_size, 1, device=device, requires_grad=False),
            )
            loss_real.backward()
            loss_fake = criterion_d(
                discriminator(shared_layers(generated_images.detach())),
                torch.zeros(batch_size, 1, device=device, requires_grad=False),
            )
            loss_fake.backward()
            optimizer_d.step()
            discriminator_loss = loss_real + loss_fake

            # train step Generator
            optimizer_g_q.zero_grad()
            shared_layers_output = shared_layers(generated_images)
            probs_fake = discriminator(shared_layers_output)
            (
                digits_estimated,
                continuous_mean,
                continuous_variance,
            ) = recognition_network(shared_layers_output)
            discrete_loss = criterion_q_discrete(
                digits_estimated,
                torch.tensor(digits, dtype=torch.int64, device=device),
            )
            continous_loss = (
                criterion_q_continous(
                    continuous_latent_code, continuous_mean, continuous_variance
                )
                * 0.1
            )
            generator_loss = (
                criterion_d(
                    probs_fake,
                    torch.ones(batch_size, 1, device=device, requires_grad=False),
                )
                + discrete_loss
                + continous_loss
            )
            generator_loss.backward()
            optimizer_g_q.step()

            if iteration == 0:
                log(
                    generator=generator,
                    epoch=epoch,
                    iteration=iteration,
                    discriminator_loss=discriminator_loss.item(),
                    generator_loss=generator_loss.item(),
                    device=device,
                    models_path=models_path,
                    figures_path=figures_path,
                )
