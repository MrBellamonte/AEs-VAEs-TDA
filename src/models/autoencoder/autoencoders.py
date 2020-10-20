import torch
from torch import nn
import numpy as np

from .base import AutoencoderModel
from .utils import View


class Autoencoder_MLP(nn.Module):
    '''
    Implementation of a "standard autoencoder"
    '''
    __slots__ = ['encoder', 'decoder']

    def __init__(self, input_dim: int, latent_dim: int, size_hidden_layers: list):
        super().__init__()


        # build encoder
        encoder = []
        encoder.append(nn.Linear(input_dim, size_hidden_layers[0]))
        encoder.append(nn.ReLU(True))
        if len(size_hidden_layers)>1:
            for i in range(0, len(size_hidden_layers)-1):
                encoder.append(nn.Linear(size_hidden_layers[i], size_hidden_layers[i+1]))
                encoder.append(nn.ReLU(True))
        encoder.append(nn.Linear(size_hidden_layers[-1], latent_dim))

        self.encoder = nn.Sequential(*encoder)

        # build decoder
        hidden_size_reversed = list(reversed(size_hidden_layers))
        decoder = []
        decoder.append(nn.Linear(latent_dim, hidden_size_reversed[0]))
        decoder.append(nn.ReLU(True))
        if len(size_hidden_layers)>1:
            for i in range(0, len(size_hidden_layers)-1):
                decoder.append(nn.Linear(hidden_size_reversed[i], hidden_size_reversed[i+1]))
                decoder.append(nn.ReLU(True))
        decoder.append(nn.Linear(hidden_size_reversed[-1], input_dim))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        z = self.encoder(input)
        x = self.decoder(z)
        return x, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class Autoencoder_MLP_topoae(AutoencoderModel):
    '''
    Implementation of a "standard autoencoder"
    '''
    __slots__ = ['encoder', 'decoder','reconst_error']

    def __init__(self, input_dim: int, latent_dim: int, size_hidden_layers: list):
        super().__init__()

        # build encoder
        encoder = []
        encoder.append(nn.Linear(input_dim, size_hidden_layers[0]))
        encoder.append(nn.ReLU(True))
        if len(size_hidden_layers) > 1:
            for i in range(0, len(size_hidden_layers)-1):
                encoder.append(nn.Linear(size_hidden_layers[i], size_hidden_layers[i+1]))
                encoder.append(nn.ReLU(True))
        encoder.append(nn.Linear(size_hidden_layers[-1], latent_dim))

        self.encoder = nn.Sequential(*encoder)

        # build decoder
        hidden_size_reversed = list(reversed(size_hidden_layers))
        decoder = []
        decoder.append(nn.Linear(latent_dim, hidden_size_reversed[0]))
        decoder.append(nn.ReLU(True))
        if len(size_hidden_layers) > 1:
            for i in range(0, len(size_hidden_layers)-1):
                decoder.append(nn.Linear(hidden_size_reversed[i], hidden_size_reversed[i+1]))
                decoder.append(nn.ReLU(True))
        decoder.append(nn.Linear(hidden_size_reversed[-1], input_dim))

        self.decoder = nn.Sequential(*decoder)

        self.reconst_error = nn.MSELoss()

    def forward(self, input):
        latent = self.encoder(input)
        x_reconst = self.decoder(latent)
        reconst_error = self.reconst_error(input, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class ConvAE_MNIST(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            View((-1, 8*2*2)),
            nn.Linear(8*2*2, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8*2*2),
            View((-1, 8, 2, 2)),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x.reshape(x.shape[0],1,28, 28))

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z).reshape(z.shape[0],28*28)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}

