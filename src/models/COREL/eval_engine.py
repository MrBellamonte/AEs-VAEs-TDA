import pickle
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from dep.topo_ae_code.src_topoae.models import TopologicallyRegularizedAutoencoder
from src.models.autoencoders import (
    Autoencoder_MLP, Autoencoder_MLP_topoae,
    Autoencoder_MLP_topoae_eval, Autoencoder_MLP_topoaeeval2)


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
        #model = Autoencoder_MLP(input_dim=101, latent_dim=2, size_hidden_layers=[32 , 64 , 32])


        autoencoder = Autoencoder_MLP_topoaeeval2(input_dim=101, latent_dim=2, size_hidden_layers=[32, 32])
        # autoencoder = Autoencoder_MLP_topoaeeval2(input_dim=101, latent_dim=2,
        #                                           size_hidden_layers=[128, 64, 32])
        model = Autoencoder_MLP_topoae_eval(autoencoder)

        path_model = path_to_folder+'models.pht'
        model.load_state_dict(torch.load(path_model))

        #model = model.autoencoder

    else:
        # get config to initialize models
        path_config = path_to_folder+'config.pickle'
        infile = open(path_config, 'rb')
        config = pickle.load(infile)
        infile.close()

        model_kwargs = config['model_kwargs']


        path_model = path_to_folder+'models.pht'

        #todo: works only for autoencoder... Fix if necessary
        model = Autoencoder_MLP(**model_kwargs['kwargs'])
        model.load_state_dict(torch.load(path_model))

    return model

def get_latentspace_representation(model, data: TensorDataset, device = 'cpu'):
    dl = DataLoader(data, batch_size=500, num_workers=4)
    X, Z, Y = [], [], []
    model.eval()
    sys.setrecursionlimit(10000)
    model.to(device)
    for x, y in dl:
        x = x.to(device)

        #x_hat, z = model(x.float())

        x_hat, z = model(x.float())
        # x_hat = model.decoder(z.float())
        # print(x_hat)
        X.append(x_hat)
        Y.append(y)
        Z.append(z)


    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    Z = torch.cat(Z, dim=0)

    return X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy(),


