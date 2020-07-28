"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import os
import pickle


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict


from dep.topo_ae_code.src_topoae.models import TopologicallyRegularizedAutoencoder
from src.models.TopoAE.config import ConfigTopoAE, ConfigGrid_TopoAE


def train_TopoAE(data: TensorDataset, config: ConfigTopoAE, root_folder, verbose = False):

    model_class = config.model_class
    autoencoder = model_class(**config.model_kwargs)

    model = TopologicallyRegularizedAutoencoder(autoencoder, lam = config.top_loss_weight)

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate)

    dl = DataLoader(data,
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=True)

    log = defaultdict(list)

    model.train()

    for epoch in range(1,config.n_epochs+1):

        for x, _ in dl:
            loss, loss_components = model(x)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log lifetimes as well as all losses we compute
            log['loss.autoencoder'].append(loss_components['loss.autoencoder'])
            log['loss.topo_error'].append(loss_components['loss.topo_error'])

        if verbose:
            print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
                epoch,
                loss_components['loss.autoencoder'].detach().numpy().mean(),
                loss_components['loss.topo_error'].detach().numpy().mean()))


    path = os.path.join(root_folder, config.creat_uuid())
    os.makedirs(path)

    config_dict = config.create_dict()
    config_dict['uuid'] = config.creat_uuid()

    # Save models
    torch.save(model.state_dict(), '.'.join([path + '/models', 'pht']))

    # Save the config used for training as well as all logging results
    #todo fix log!
    out_data = [config_dict, log]
    file_ext = ['config', 'log']
    for x, y in zip(out_data, file_ext):
        with open('.'.join([path + '/'+ y, 'pickle']), 'wb') as fid:
            pickle.dump(x, fid)

    #todo calculate and save metrics after training


def simulator_TopoAE(config_grid: ConfigGrid_TopoAE, path: str, verbose: bool = False, data_constant: bool = False):

    if verbose:
        print('Load and verify configurations...')
    configs = config_grid.configs_from_grid()
    for config in configs:
        config.check()


    if data_constant:
        print('WARNING: Model runs with same data for all configurations!')
        # sample data
        if verbose:
            print('Sample data...')

        dataset = configs[0].dataset
        X, y = dataset.sample(**configs[0].sampling_kwargs)
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))



    if verbose:
        print('START!')
    # train models for all configurations


    for i, config in enumerate(configs):
        print('Configuration {} out of {}'.format(i+1, len(configs)))
        if not data_constant:
            # sample data
            if verbose:
                print('Sample data...')

            dataset = config.dataset
            X, y = dataset.sample(**config.sampling_kwargs)
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))

        if verbose:
            print('Run model...')


        train_TopoAE(dataset, config, path, verbose = verbose)





