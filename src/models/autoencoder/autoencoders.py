import torch
from torch import nn

from .base import AutoencoderModel


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
        return self.decode(z)


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

class Autoencoder_MLP_topoae_eval(nn.Module):
    '''
    Implementation of a "standard autoencoder"
    '''
    __slots__ = ['autoencoder','encoder', 'decoder','latent_norm']

    def __init__(self, autoencoder):
        super().__init__()

        self.autoencoder = autoencoder
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)


    def forward(self, input):
        self.autoencoder.eval()
        z = self.autoencoder.encoder(input)
        x = self.autoencoder.decoder(z)
        return x, z




class Autoencoder_MLP_topoaeeval2(nn.Module):
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
        pass


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)