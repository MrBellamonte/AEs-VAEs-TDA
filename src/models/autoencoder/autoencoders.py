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


class ConvAE_MNIST_3D(AutoencoderModel):
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
            nn.Linear(16, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
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

class ConvAE_MNIST_4D(AutoencoderModel):
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
            nn.Linear(16, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
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

class ConvAE_MNIST_8D(AutoencoderModel):
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
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
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


class DeepAE_MNIST(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 784),
            nn.ReLU(True),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

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

class DeepAE_MNIST_3D(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 784),
            nn.ReLU(True),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

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


class DeepAE_MNIST_4D(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 784),
            nn.ReLU(True),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

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


class DeepAE_MNIST_8D(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 784),
            nn.ReLU(True),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

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


class ConvAE_MNIST_SMALL(AutoencoderModel):
    """Convolutional Autoencoder for MNIST"""

    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
        self.conv4 = nn.MaxPool2d(2, stride=2)

        self.conv4t = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        self.conv3t = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        self.conv2t = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=2, stride=2)
        self.conv1t = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=1,
                                    padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            self.conv2,
            nn.ReLU(True),
            self.conv3,
            nn.ReLU(True),
            self.conv4,
            nn.ReLU(True),
            View((-1, 8*6*6)),
            nn.Linear(8*6*6, 144),
            nn.ReLU(True),
            nn.Linear(144, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, 144),
            nn.ReLU(True),
            nn.Linear(144, 8*6*6),
            View((-1, 8, 6, 6)),
            self.conv4t,
            nn.ReLU(True),
            self.conv3t,
            nn.ReLU(True),
            self.conv2t,
            nn.ReLU(True),
            self.conv1t,
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


class ConvAE_Unity480320(AutoencoderModel):
    """Convolutional Autoencoder for unity data large"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4)

        self.conv5t = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4)
        self.conv4t = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv3t = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.conv2t = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2)
        self.conv1t = nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=1, stride=1)


        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            self.conv2,
            nn.ReLU(True),
            self.conv3,
            nn.ReLU(True),
            self.conv4,
            nn.ReLU(True),
            self.conv5,
            nn.ReLU(True),
            View((-1, 64*15*10)),
            nn.Linear(64*15*10, 8),
            nn.ReLU(True),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 64*15*10),
            View((-1, 64,10,15)),
            self.conv5t,
            nn.ReLU(True),
            self.conv4t,
            nn.ReLU(True),
            self.conv3t,
            nn.ReLU(True),
            self.conv2t,
            nn.ReLU(True),
            self.conv1t,
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()
    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

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


class ConvAE_Unity480320_inference(AutoencoderModel):
    """Convolutional Autoencoder for unity data large"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4)

        self.conv5t = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4)
        self.conv4t = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv3t = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.conv2t = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2)
        self.conv1t = nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=1, stride=1)


        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            self.conv2,
            nn.ReLU(True),
            self.conv3,
            nn.ReLU(True),
            self.conv4,
            nn.ReLU(True),
            self.conv5,
            nn.ReLU(True),
            View((-1, 64*15*10)),
            nn.Linear(64*15*10, 8),
            nn.ReLU(True),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 64*15*10),
            View((-1, 64,10,15)),
            self.conv5t,
            nn.ReLU(True),
            self.conv4t,
            nn.ReLU(True),
            self.conv3t,
            nn.ReLU(True),
            self.conv2t,
            nn.ReLU(True),
            self.conv1t,
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()
    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        return self.encode(x)
