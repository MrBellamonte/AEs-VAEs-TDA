"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import operator
import os

import pandas as pd

import torch
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import FileStorageObserver
from torch.utils.data import TensorDataset

from scripts.ssc.models.vanillaAE.config_libraries.debug import ae_test
from src.models.TopoAE.config import ConfigTopoAE
from src.train_pipeline.sacred_observer import SetID

from src.train_pipeline.train_model import train

SETTINGS['CAPTURE_MODE'] = 'sys'
ex = Experiment()
COLS_DF_RESULT = list(ae_test.configs_from_grid()[0].create_id_dict().keys())+['metric', 'value']


@ex.config
def cfg():
    config = ae_test.configs_from_grid()[0]
    experiment_dir = '~/'
    experiment_root = '~/'
    seed = 0
    device = 'cpu'
    num_threads = 1
    verbose = False


@ex.automain
def train_VanillaAE(_run, _seed, _rnd, config: ConfigTopoAE, experiment_dir, experiment_root, device,
                 num_threads, verbose):
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
    X_train, y_train = dataset.sample(**config.sampling_kwargs, train=True)
    dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    X_test, y_test = dataset.sample(**config.sampling_kwargs, train=False)
    dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    torch.manual_seed(_seed)
    if device == 'cpu' and num_threads is not None:
        torch.set_num_threads(num_threads)

    # Initialize model
    model_class = config.model_class
    autoencoder = model_class(**config.model_kwargs)

    model = autoencoder

    model.to(device)

    # Train and evaluate model
    result = train(model=model, data_train=dataset_train, data_test=dataset_test, config=config,
                   device=device, quiet=operator.not_(verbose), val_size=0.2, _seed=_seed,
                   _rnd=_rnd, _run=_run, rundir=experiment_dir)

    # Format experiment data
    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df.columns = ['metric', 'value']

    id_dict = config.create_id_dict()
    for key, value in id_dict.items():
        df[key] = value
    df.set_index('uid')

    df = df[COLS_DF_RESULT]

    df.to_csv(os.path.join(experiment_root, 'eval_metrics_all.csv'), mode='a', header=False)

def simulator_VanillaAE(config):
    id = config.creat_uuid()
    try:
        ex.observers[0] = SetID(id)
        ex.observers[1] = FileStorageObserver(config.experiment_dir)
    except:
        ex.observers.append(SetID(id))
        ex.observers.append(FileStorageObserver(config.experiment_dir))

    ex_dir_new = os.path.join(config.experiment_dir, id)

    ex.run(config_updates={'config'         : config, 'experiment_dir': ex_dir_new,
                           'experiment_root': config.experiment_dir,
                           'seed'           : config.seed, 'device': config.device,
                           'num_threads'    : config.num_threads,
                           'verbose'        : config.verbose
                           })
