from torch import nn


class autoencoder(nn.Module):
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
