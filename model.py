"""
Fully connected layers have been replaced with conv layers.
Inspired by: https://github.com/pianomania/infoGAN-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLayers(nn.Module):
    """Shared by Discriminator D and RecognitionNetwork Q"""

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=1024,
                kernel_size=7,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.sequential(x)


class Discriminator(nn.Module):
    """
    This is the part of the discriminator that is not shared with the
    recognition network.  It is expected to run more or less like this:
    discriminator(shared_layers(x))
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x)).squeeze(-1).squeeze(-1)


class RecognitionNetwork(nn.Module):
    """
    a.k.a. Q
    This is the part of the recognition network that is not shared with the
    discriminator.  It is expected to run more or less like this:
    recognition_network(shared_layers(x)) Output is the estimation of the
    categorical latent variable, and a mean and variance for each continuos
    latent variable, in this case 2.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1024,
            out_channels=128,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=128)
        self.conv_categorical = nn.Conv2d(
            in_channels=128,
            out_channels=10,
            kernel_size=1,
        )
        self.conv_mean = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
        )
        self.conv_var = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
        )

    def forward(self, x):
        x = F.leaky_relu(
            self.batch_norm(self.conv(x)),
            negative_slope=0.1,
            inplace=True,
        )
        digits_estimate = self.conv_categorical(x).squeeze()
        mean = self.conv_mean(x).squeeze()
        # "exponential transformation of the network output to ensure positivity"
        variance = self.conv_var(x).squeeze().exp()
        return digits_estimate, mean, variance


class Generator(nn.Module):
    """
    The generator uses 2D transposed convolution and batch norm.
    No bias needed because of the batch norm.
    No shared layers with other networks, nothing really special.
    """

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=74,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=128,
                kernel_size=7,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass. The input is expected to be a concatenation of the noise z
        and the latent codes, e.g. like this:
        generator(
            torch.cat([noise, discrete_latent_code, continuous_latent_code], dim=1)
        )
        """
        return self.seq(x.unsqueeze(-1).unsqueeze(-1))
