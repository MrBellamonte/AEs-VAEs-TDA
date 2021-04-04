import torch
from torch import nn

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
