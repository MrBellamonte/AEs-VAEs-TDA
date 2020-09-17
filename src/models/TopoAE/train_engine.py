"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import operator
import os

import pandas as pd

import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.manifold import SpectralEmbedding
from torch.optim import optimizer
from torch.utils.data import TensorDataset, DataLoader

from scripts.ssc.TopoAE.topoae_config_library import placeholder_config_topoae
from src.models.COREL.eval_engine import get_latentspace_representation
from src.models.TopoAE.approx_based import TopologicallyRegularizedAutoencoder
from src.models.TopoAE.config import ConfigTopoAE, ConfigGrid_TopoAE
from src.train_pipeline.sacred_observer import SetID

from src.train_pipeline.train_model import train
from src.utils.plots import plot_classes_qual

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


    # Set data sampling seed
    if 'seed' in config.sampling_kwargs:
        seed_sampling = config.sampling_kwargs['seed']
    else:
        seed_sampling = _seed


    # Sample data
    dataset = config.dataset
    X_train, y_train = dataset.sample(**config.sampling_kwargs, seed=seed_sampling, train=True)
    dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    X_test, y_test = dataset.sample(**config.sampling_kwargs, seed=seed_sampling, train=False)
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
    if config.method_args['LLE_pretrain']:
        print('Compute LLE')
        embedding = SpectralEmbedding(n_components=2, n_jobs=num_threads, n_neighbors=90)
        X_transformed = embedding.fit_transform(X_train)
        plot_classes_qual(X_transformed, y_train, path_to_save=None, title=None, show=True)
        dataset_visualisation = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        visualisation_loader = DataLoader(dataset_visualisation, batch_size=config.batch_size,
                                          shuffle=True,
                                          pin_memory=True, drop_last=True)
        X_train, Y_train, Z_train = get_latentspace_representation(model, visualisation_loader,
                                                                   device=device)
        plot_classes_qual(Z_train, Y_train, path_to_save=None, title=None, show=True)
        dataset_pretrain = TensorDataset(torch.Tensor(X_train),torch.Tensor(X_transformed), torch.Tensor(y_train))
        print('Done!')
        train_loader = DataLoader(dataset_pretrain, batch_size=config.batch_size, shuffle=False,
                                  pin_memory=True, drop_last=False)
        run_times_epoch = []
        print('Start Pretraining')
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate)
        for epoch in range(1, 40):
            for batch, (data, latent_LLE, label) in enumerate(train_loader):

                x = data.to('cpu')
                latent = model.encode(x)
                x_rec = model.decode(latent)

                latent_loss = torch.nn.MSELoss()
                rec_loss = torch.nn.MSELoss()

                loss = latent_loss(latent, latent_LLE) + rec_loss(x, x_rec)
                print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dataset_visualisation = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        visualisation_loader = DataLoader(dataset_visualisation, batch_size=config.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=False)
        X_train, Y_train, Z_train = get_latentspace_representation(model, visualisation_loader,
                                                                   device=device)
        plot_classes_qual(Z_train, Y_train, path_to_save=None, title=None, show=True)
        print('Done')




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

