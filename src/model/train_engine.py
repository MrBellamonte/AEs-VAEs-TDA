"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import os
import pickle


import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict

from torchph.pershom import pershom_backend

from src.utils.config_utils import (check_config, configs_from_grid, create_model_uuid, create_data_uuid)

vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1

# config
DEVICE  = "cuda"



def train(data: TensorDataset, config, root_folder, data_uuid = '', verbose = False):

    # HARD-CODED conifg
    # todo parametrize as well
    ball_radius = 1.0 #only affects the scaling



    train_args = config['train_args']
    model_args = config['model_args']

    model_class = model_args['model_class']

    model = model_class(**model_args['kwargs']).to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=train_args['learning_rate'])

    dl = DataLoader(data,
                    batch_size=train_args['batch_size'],
                    shuffle=True,
                    drop_last=True)

    log = defaultdict(list)

    model.train()

    for epoch in range(1,train_args['n_epochs']+1):

        for x, _ in dl:
            x = x.to(DEVICE)

            # Get reconstruction x_hat and latent
            # space representation z
            x_hat, z = model(x.float())

            # Set both losses to 0 in case we ever want to
            # disable one and still use the same logging code.
            top_loss = torch.tensor([0]).type_as(x_hat)

            # Computes l1-reconstruction loss
            rec_loss_func = train_args['rec_loss']['loss_class']
            rec_loss = rec_loss_func(x_hat, x)

            # For each branch in the latent space representation,
            # we enforce the topology loss and track the lifetimes
            # for further analysis.
            lifetimes = []
            pers = vr_l1_persistence(z[:,:].contiguous(), 0, 0)[0][0]

            if pers.dim() == 2:
                pers = pers[:, 1]
                lifetimes.append(pers.tolist())
                top_loss_func = train_args['top_loss']['loss_class']
                top_loss +=top_loss_func(pers,2.0*ball_radius*torch.ones_like(pers))

            # Log lifetimes as well as all losses we compute
            log['lifetimes'].append(lifetimes)
            log['top_loss'].append(top_loss.item())
            log['rec_loss'].append(rec_loss.item())

            loss = train_args['rec_loss']['weight']*rec_loss + train_args['top_loss']['weight']*top_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
                epoch,
                np.array(log['rec_loss'][-int(len(data)/train_args['batch_size']):]).mean()*train_args['rec_loss']['weight'],
                np.array(log['top_loss'][-int(len(data)/train_args['batch_size']):]).mean()*train_args['top_loss']['weight']))

    # Create a unique base filename
    the_uuid = data_uuid + create_model_uuid(config)

    path = os.path.join(root_folder, the_uuid)
    os.makedirs(path)
    config['uuid'] = the_uuid

    # Save model
    torch.save(model.state_dict(), '.'.join([path + '/model', 'pht']))


    # Save the config used for training as well as all logging results
    out_data = [config, log]
    file_ext = ['config', 'log']
    for x, y in zip(out_data, file_ext):
        with open('.'.join([path + '/'+ y, 'pickle']), 'wb') as fid:
            pickle.dump(x, fid)


def simulator(config_grid, path, verbose = False, create_datauuid = True):

    # sample data
    if verbose:
        print('Sample data...')
    data_args = config_grid.pop('data_args')

    dataset = data_args['dataset']
    X, y = dataset.sample(**data_args['kwargs'])
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    if create_datauuid:
        data_uuid = create_data_uuid(data_args) + '-'

    if verbose:
        print('Load and verify configurations...')
    configs = configs_from_grid(config_grid)
    for config in configs:
        check_config(config)

    if verbose:
        print('START!')
    # train model for all configurations


    for i, config in enumerate(configs):
        print('Run model for configuration {} out of {}'.format(i+1, len(configs)))
        torch.cuda.empty_cache()
        train(dataset, config, path, data_uuid, verbose = verbose)





