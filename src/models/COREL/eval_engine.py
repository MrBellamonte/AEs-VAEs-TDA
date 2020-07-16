import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoders import autoencoder


def get_config(path_to_folder):
    path = path_to_folder+'config.pickle'
    infile = open(path, 'rb')
    config = pickle.load(infile)
    infile.close()

    return config


def get_log(path_to_folder):
    path = path_to_folder+'log.pickle'
    infile = open(path, 'rb')
    log = pickle.load(infile)
    infile.close()

    return log

def get_model(path_to_folder, config_fix = False):
    '''
    config_fix: allows to hardcode model in case configuration file is corrupted
    '''
    if config_fix:
        model = autoencoder(input_dim=101, latent_dim=2, size_hidden_layers=[128 ,64 ,32])

        path_model = path_to_folder+'model.pht'
        model.load_state_dict(torch.load(path_model))

    else:
        # get config to initialize models
        path_config = path_to_folder+'config.pickle'
        infile = open(path_config, 'rb')
        config = pickle.load(infile)
        infile.close()

        model_kwargs = config['model_kwargs']


        path_model = path_to_folder+'models.pht'

        #todo: works only for autoencoder... Fix if necessary
        model = autoencoder(**model_kwargs['kwargs'])
        model.load_state_dict(torch.load(path_model))

    return model

def get_latentspace_representation(model, data: TensorDataset, device = 'cpu'):
    dl = DataLoader(data, batch_size=100, num_workers=4)
    X, Z, Y = [], [], []
    model.eval()
    model.to(device)
    for x, y in dl:
        x = x.to(device)

        x_hat, z = model(x.float())
        X.append(x_hat)
        Y.append(y)
        Z.append(z)

    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    Z = torch.cat(Z, dim=0)

    return X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy(),


