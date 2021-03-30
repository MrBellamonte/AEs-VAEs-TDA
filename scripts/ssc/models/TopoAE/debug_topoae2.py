import operator
import os

import pandas as pd

import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.data import TensorDataset

from scripts.ssc.models.TopoAE.config_libraries.local_configs.mnist2 import mnist_test_loc
from scripts.ssc.models.TopoAE.topoae_config_library import placeholder_config_topoae
from src.models.TopoAE.approx_based import TopologicallyRegularizedAutoencoder
from src.models.TopoAE.config import ConfigTopoAE
from src.train_pipeline.sacred_observer import SetID

from src.train_pipeline.train_model import train


def train_TopoAE(_seed, config: ConfigTopoAE, experiment_dir, experiment_root, device,
                 num_threads, verbose):
    try:
        os.makedirs(experiment_dir)
    except:
        pass

    try:
        os.makedirs(experiment_root)
    except:
        pass



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

    model = TopologicallyRegularizedAutoencoder(autoencoder, lam_r=config.rec_loss_weight,
                                                lam_t=config.top_loss_weight,
                                                toposig_kwargs=config.toposig_kwargs)
    model.to(device)

    # Train and evaluate model
    result = train(model=model, data_train=dataset_train, data_test=dataset_test, config=config,
                   device=device, quiet=operator.not_(verbose), val_size=0.2, _seed=_seed,
                   _rnd=_rnd, _run=_run, rundir=experiment_dir)

    # Format experiment data



if __name__ == "__main__":
    train_TopoAE(mnist_test_loc.configs_from_grid()[0])