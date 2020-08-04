"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import operator
import os
import pickle
import time

import pandas as pd
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset



from scripts.ssc.TopoAE.topoae_config_library import placeholder_config_topoae
from src.models.TopoAE.approx_based import TopologicallyRegularizedAutoencoder
from src.models.TopoAE.config import ConfigTopoAE, ConfigGrid_TopoAE
from src.train_pipeline.sacred_observer import SetID

from src.train_pipeline.train_model import train

ex = Experiment()
COLS_DF_RESULT = list(placeholder_config_topoae.create_id_dict().keys())+['metric', 'value']


@ex.config
def cfg():
    config = placeholder_config_topoae
    experiment_dir = '~/'
    experiment_root = '~/'
    seed = 0
    device = 'cpu'
    num_threads = 1
    verbose = False


@ex.automain
def train_TopoAE(_run, _seed, _rnd, config: ConfigTopoAE, experiment_dir, experiment_root, device, num_threads, verbose):

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

    # Sample data
    dataset = config.dataset
    X_train, y_train = dataset.sample(**config.sampling_kwargs, seed=_seed, train=True)
    dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    X_test, y_test = dataset.sample(**config.sampling_kwargs, seed=_seed, train=False)
    dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    torch.manual_seed(_seed)
    if device == 'cpu' and num_threads is not None:
        torch.set_num_threads(num_threads)


    # Initialize model
    model_class = config.model_class
    autoencoder = model_class(**config.model_kwargs)
    model = TopologicallyRegularizedAutoencoder(autoencoder, lam_r=config.rec_loss_weight, lam_t=config.top_loss_weight,
                                                toposig_kwargs=config.toposig_kwargs)
    model.to(device)

    # Train and evaluate model
    result = train(model = model, data_train = dataset_train, data_test = dataset_test, config = config, device = device, quiet = operator.not_(verbose), val_size = 0.2, _seed = _seed,
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



def simulator_TopoAE(config_grid: ConfigGrid_TopoAE):

    ex.observers.append(FileStorageObserver(config_grid.experiment_dir))
    ex.observers.append(SetID('myid'))

    for config in config_grid.configs_from_grid():
        id = config.creat_uuid()
        ex_dir_new = os.path.join(config_grid.experiment_dir, id)
        ex.observers[1] = SetID(id)
        ex.run(config_updates={'config': config, 'experiment_dir' : ex_dir_new, 'experiment_root' : config_grid.experiment_dir,
                               'seed' : config_grid.seed, 'device' : config_grid.device, 'num_threads' : config_grid.num_threads,
                               'verbose' : config_grid.verbose
                               })





######
#
#
# def train_TopoAE(data_train: TensorDataset, data_test: TensorDataset, config: ConfigTopoAE, root_folder, verbose = False, num_threads = None):
#
#     if num_threads is not None:
#         torch.set_num_threads(num_threads)
#
#     model_class = config.model_class
#     autoencoder = model_class(**config.model_kwargs)
#
#     model = TopologicallyRegularizedAutoencoder(autoencoder, lam = config.top_loss_weight, toposig_kwargs=config.toposig_kwargs)
#
#     optimizer = Adam(
#         model.parameters(),
#         lr=config.learning_rate,
#         weight_decay=config.weight_decay
#     )
#
#     train_dataset, validation_dataset = split_validation(
#         data_train, config.val_size, config.split_seed)
#
#     dl = DataLoader(train_dataset,
#                     batch_size=config.batch_size,
#                     shuffle=True,
#                     drop_last=True)
#
#     log = defaultdict(list)
#
#     model.train()
#
#     df_log = pd.DataFrame()
#
#     for epoch in range(1,config.n_epochs+1):
#         t0 = time.time()
#         for x, _ in dl:
#
#             loss, loss_components = model(x)
#
#             # Optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Log lifetimes as well as all losses we compute
#             log['loss.autoencoder'].append(loss_components['loss.autoencoder'])
#             log['loss.topo_error'].append(loss_components['loss.topo_error'])
#
#
#         if verbose:
#             print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
#                 epoch,
#                 loss_components['loss.autoencoder'].detach().numpy().mean(),
#                 loss_components['loss.topo_error'].detach().numpy().mean()))
#         t1 = time.time()
#         row = {str(epoch): dict(rec_loss=loss_components['loss.autoencoder'].detach().numpy().mean(),
#                                 topo_loss=loss_components['loss.topo_error'].detach().numpy().mean(),
#                                 time_epoch = (t1-t0))}
#         df_log = df_log.append(pd.DataFrame.from_dict(row, orient='index'))
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
#
#     df_log.to_csv(path+'/df_logs.csv')
#
#     for x, y in zip(out_data, file_ext):
#         with open('.'.join([path + '/'+ y, 'pickle']), 'wb') as fid:
#             pickle.dump(x, fid)
#
#     #todo calculate and save metrics after training
#
#
# def simulator_TopoAE(config_grid: ConfigGrid_TopoAE, path: str, verbose: bool = False, data_constant: bool = False, num_threads = None):
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
#
#         train_TopoAE(dataset, config, path, verbose = verbose, num_threads = num_threads)





