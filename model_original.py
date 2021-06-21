"""Network architecture as described in Table 1 of the paper.
- TODO: this does not work yet, training does not converge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLayers(nn.Module):
    """Shared by Discriminator D and RecognitionNetwork Q"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
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
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm2d(num_features=128),
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=128 * 7 * 7,
                out_features=1024,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(num_features=1024),
        )

    def forward(self, x):
        return self.fc(self.conv(x).view(-1, 128 * 7 * 7))


class Discriminator(nn.Module):
    """
    This is the part of the discriminator that is not shared with the
    recognition network.  It is expected to run more or less like this:
    discriminator(shared_layers(x))
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(
            in_features=1024,
            out_features=1,
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


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
        self.fc = nn.Linear(
            in_features=1024,
            out_features=128,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=128)
        self.fc_categorical = nn.Linear(
            in_features=128,
            out_features=10,
        )
        self.fc_mean = nn.Linear(
            in_features=128,
            out_features=2,
        )
        self.fc_var = nn.Linear(
            in_features=128,
            out_features=2,
        )

    def forward(self, x):
        x = F.leaky_relu(
            self.batch_norm(self.fc(x)),
            negative_slope=0.1,
            inplace=True,
        )
        digits_estimate = self.fc_categorical(x)
        mean = self.fc_mean(x)
        # "exponential transformation of the network output to ensure positivity"
        variance = self.fc_var(x).exp()
        return digits_estimate, mean, variance


class Generator(nn.Module):
    """
    The generator uses 2D transposed convolution and batch norm.
    No bias needed because of the batch norm.
    No shared layers with other networks, nothing really special.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=74,
                out_features=1024,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(
                in_features=1024,
                out_features=7 * 7 * 128,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=7 * 7 * 128),
        )
        self.conv = nn.Sequential(
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
        return self.conv(self.fc(x).view(-1, 128, 7, 7))
