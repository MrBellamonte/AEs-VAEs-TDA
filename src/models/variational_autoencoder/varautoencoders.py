import torch
from torch import nn

from src.models.autoencoder.utils import View
from src.models.variational_autoencoder.base import VariationalAutoencoderModel


class VanillaVAE(VariationalAutoencoderModel):
    '''
    Implementation of a "standard autoencoder"
    '''
    __slots__ = ['encoder', 'decoder','reconst_error']

    def __init__(self, input_dim: int, latent_dim: int, size_hidden_layers: list,lambda_kld = 1):
        super().__init__()

        # build encoder
        encoder = []
        encoder.append(nn.Linear(input_dim, size_hidden_layers[0]))
        encoder.append(nn.ReLU(True))
        if len(size_hidden_layers) > 1:
            for i in range(0, len(size_hidden_layers)-1):
                encoder.append(nn.Linear(size_hidden_layers[i], size_hidden_layers[i+1]))
                encoder.append(nn.ReLU(True))

        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(size_hidden_layers[-1], latent_dim)
        self.fc_var = nn.Linear(size_hidden_layers[-1], latent_dim)

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
        self.lambda_kld = lambda_kld


    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return self.reparameterize(mu, log_var)

    def encode_mustd(self,x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def sample(self,
               num_samples:int,
               device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def forward(self,x):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        mu, log_var = self.encode_mustd(x)
        z = self.reparameterize(mu, log_var)

        x_reconst = self.decode(z)


        reconst_error = self.reconst_error(x, x_reconst)

        kld_loss = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)

        loss = reconst_error+self.lambda_kld*kld_loss
        return loss, {'reconstruction_error': reconst_error, 'KLD': -kld_loss}

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)


class ConvVAE_Unity480320(VariationalAutoencoderModel):
    '''
    Implementation of a "standard autoencoder"
    '''
    __slots__ = ['encoder', 'decoder','reconst_error']

    def __init__(self,lambda_kld = 1):
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
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 64*15*10),
            View((-1, 64, 10, 15)),
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

        self.fc_mu = nn.Linear(8, 2)
        self.fc_var = nn.Linear(8, 2)

        self.reconst_error = nn.MSELoss()
        self.lambda_kld = lambda_kld


    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return self.reparameterize(mu, log_var)

    def encode_mustd(self,x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def sample(self,
               num_samples:int,
               device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def forward(self,x):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        mu, log_var = self.encode_mustd(x)
        z = self.reparameterize(mu, log_var)

        x_reconst = self.decode(z)


        reconst_error = self.reconst_error(x, x_reconst)

        kld_loss = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)

        loss = reconst_error+self.lambda_kld*kld_loss
        return loss, {'reconstruction_error': reconst_error, 'KLD': -kld_loss}

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)



