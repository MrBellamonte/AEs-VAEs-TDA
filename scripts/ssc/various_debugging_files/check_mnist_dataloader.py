import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.data_preprocessing.witness_complex_offline.wc_offline_utils import fetch_data
from src.datasets.datasets import MNIST
from src.models.autoencoder.autoencoders import ConvAE_MNIST_NEW, ConvAE_MNIST_SMALL

path_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2'

dataloader, landmark_distances = fetch_data(path_to_data=path_to_data)
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=784, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=784
        )
        self.reconst_error = nn.MSELoss()

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)

        reconst_error = self.reconst_error(features, reconstructed)
        return reconst_error, {'reconstruction_error': reconst_error}

class AE2(nn.Module):
    """Convolutional Autoencoder for unity data large"""

    def __init__(self):
        super().__init__()


        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
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

class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder for MNIST/Fashion MNIST."""

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
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

if __name__ == "__main__":
    model = ConvAE_MNIST_SMALL()
    #model = AE()
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=5,
        weight_decay=0.000001)
    # in loop

    dataset = MNIST()
    data,labels = dataset.sample()

    print(data.shape)
    print(data.max())
    data = data[:(128*120)]/255

    data_torch = torch.Tensor(data)
    data_torch = TensorDataset(data_torch,data_torch)
    dataloader = DataLoader(
            data_torch, batch_size=64, pin_memory=True, drop_last=True,shuffle=False)

    losses=0
    for epoch in range(1000):
        print('EPOCH: {}'.format(epoch))
        for bs_i, (data,labels) in enumerate(dataloader):
            model.train()
            if bs_i < 40:
                loss, loss_components = model(data)
                losses +=float(loss_components['reconstruction_error'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        print(losses/40)
        losses = 0

