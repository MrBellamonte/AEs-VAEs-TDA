"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import operator
import os
import pickle


import torch
import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict

from torchph.pershom import pershom_backend

from scripts.ssc.COREL.config_library import placeholder_config_corel
from src.models.COREL.COREL_AE import COREL_H0_Autoencoder
from src.models.COREL.config import ConfigCOREL, ConfigGrid_COREL
from src.train_pipeline.sacred_observer import SetID
from src.train_pipeline.train_model import train

vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1

# config
DEVICE  = "cuda"

ex = Experiment()
COLS_DF_RESULT = list(placeholder_config_corel.create_id_dict().keys())+['metric', 'value']


@ex.config
def cfg():
    config = placeholder_config_corel
    experiment_dir = '~/'
    experiment_root = '~/'
    seed = 0
    device = DEVICE
    verbose = False





@ex.automain
def train_COREL_2(_run, _seed, _rnd, config: ConfigCOREL, experiment_dir, experiment_root, verbose):

    try:
        os.makedirs(experiment_dir)
    except:
        pass

    try:
        os.makedirs(experiment_root)
    except:
        pass

    if os.path.isfile(os.path.join(experiment_root, 'eval_metrics_all.csv')):
        pass
    else:
        df = pd.DataFrame(columns=COLS_DF_RESULT)
        df.to_csv(os.path.join(experiment_root, 'eval_metrics_all.csv'))

    dataset = config.dataset
    X_train, y_train = dataset.sample(**config.sampling_kwargs, seed=_seed, train=True)
    dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    X_test, y_test = dataset.sample(**config.sampling_kwargs, seed=_seed, train=False)
    dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    torch.manual_seed(_seed)


    # Initialize model
    model_class = config.model_class
    autoencoder = model_class(**config.model_kwargs).to(DEVICE)

    model = COREL_H0_Autoencoder(autoencoder,
                                 rec_loss_func=config.rec_loss,
                                 top_loss_func=config.top_loss,
                                 lam_r=config.rec_loss_weight,
                                 lam_t=config.top_loss_weight
                                 )

    # Train and evaluate model
    result = train(model = model, data_train = dataset_train, data_test = dataset_test, config = config, device = DEVICE, quiet = operator.not_(verbose), val_size = 0.2, _seed = _seed,
          _rnd = _rnd, _run = _run, rundir = experiment_dir)


    # Format experiment data
    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df.columns = ['metric', 'value']

    id_dict = config.create_id_dict()
    for key, value in id_dict.items():
        df[key] = value
    df.set_index('uid')

    df = df[COLS_DF_RESULT]

    df.to_csv(os.path.join(experiment_root, 'eval_metrics_all.csv'), mode='a', header=False)


def simulator_COREL_2(config_grid: ConfigGrid_COREL):

    ex.observers.append(FileStorageObserver(config_grid.experiment_dir))
    ex.observers.append(SetID('myid'))

    for config in config_grid.configs_from_grid():
        id = config.creat_uuid()
        ex_dir_new = os.path.join(config_grid.experiment_dir, id)
        ex.observers[1] = SetID(id)
        ex.run(config_updates={'config': config, 'experiment_dir' : ex_dir_new, 'experiment_root' : config_grid.experiment_dir,
                               'seed' : config_grid.seed, 'verbose' : config_grid.verbose
                               })




# def train_COREL(data: TensorDataset, config: ConfigCOREL, root_folder, verbose = False):
#
#     # HARD-CODED conifg
#     # todo parametrize as well
#     ball_radius = 1.0 #only affects the scaling
#
#
#     model_class = config.model_class
#
#     model = model_class(**config.model_kwargs).to(DEVICE)
#
#     optimizer = Adam(
#         model.parameters(),
#         lr=config.learning_rate)
#
#     dl = DataLoader(data,
#                     batch_size=config.batch_size,
#                     shuffle=True,
#                     drop_last=True)
#
#     log = defaultdict(list)
#
#     model.train()
#
#     for epoch in range(1,config.n_epochs+1):
#
#         for x, _ in dl:
#             x = x.to(DEVICE)
#
#             # Get reconstruction x_hat and latent
#             # space representation z
#             x_hat, z = model(x.float())
#
#             # Set both losses to 0 in case we ever want to
#             # disable one and still use the same logging code.
#             top_loss = torch.tensor([0]).type_as(x_hat)
#
#             # Computes l1-reconstruction loss
#             rec_loss_func = config.rec_loss
#             rec_loss = rec_loss_func(x_hat, x)
#
#             # For each branch in the latent space representation,
#             # we enforce the topology loss and track the lifetimes
#             # for further analysis.
#             lifetimes = []
#             pers = vr_l1_persistence(z[:,:].contiguous(), 0, 0)[0][0]
#
#             if pers.dim() == 2:
#                 pers = pers[:, 1]
#                 lifetimes.append(pers.tolist())
#                 top_loss_func = config.top_loss
#                 top_loss +=top_loss_func.forward(pers,2.0*ball_radius*torch.ones_like(pers))
#
#             # Log lifetimes as well as all losses we compute
#             log['lifetimes'].append(lifetimes)
#             log['top_loss'].append(top_loss.item())
#             log['rec_loss'].append(rec_loss.item())
#
#             loss = config.rec_loss_weight*rec_loss + config.top_loss_weight*top_loss
#
#             model.zero_grad()
#             loss.backward()
#             optimizer.step()
#         if verbose:
#             print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
#                 epoch,
#                 np.array(log['rec_loss'][-int(len(data)):]).mean()*config.rec_loss_weight,
#                 np.array(log['top_loss'][-int(len(data)):]).mean()*config.top_loss_weight))
#
#
#     path = os.path.join(root_folder, config.creat_uuid())
#     os.makedirs(path)
#
#     config_dict = config.create_dict()
#     config_dict['uuid'] = config.creat_uuid()
#
#     # Save models
#     torch.save(model.state_dict(), '.'.join([path + '/models', 'pht']))
#
#     # Save the config used for training as well as all logging results
#     #todo fix log!
#     out_data = [config_dict, log]
#     file_ext = ['config', 'log']
#     for x, y in zip(out_data, file_ext):
#         with open('.'.join([path + '/'+ y, 'pickle']), 'wb') as fid:
#             pickle.dump(x, fid)
#
#     #todo calculate and save metrics after training
#
#
# def simulator_COREL(config_grid: ConfigGrid_COREL, path: str, verbose: bool = False, data_constant: bool = False):
#
#     if verbose:
#         print('Load and verify configurations...')
#     configs = config_grid.configs_from_grid()
#     for config in configs:
#         config.check()
#
#
#     if data_constant:
#         print('WARNING: Model runs with same data for all configurations!')
#         # sample data
#         if verbose:
#             print('Sample data...')
#
#         dataset = configs[0].dataset
#         X, y = dataset.sample(**configs[0].sampling_kwargs)
#         dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
#
#
#
#     if verbose:
#         print('START!')
#     # train models for all configurations
#
#
#     for i, config in enumerate(configs):
#         print('Configuration {} out of {}'.format(i+1, len(configs)))
#         if not data_constant:
#             # sample data
#             if verbose:
#                 print('Sample data...')
#
#             dataset = config.dataset
#             X, y = dataset.sample(**config.sampling_kwargs)
#             dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
#
#         if verbose:
#             print('Run model...')
#
#         torch.cuda.empty_cache()
#         train_COREL(dataset, config, path, verbose = verbose)
#
#
#
#
#
