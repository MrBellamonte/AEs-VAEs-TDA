"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import operator
import os

import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver

from src.competitors.config import (
    ConfigGrid_Competitors, Config_Competitors,
    placeholder_config_competitors)
from src.competitors.train_competitor import train_comp

from src.train_pipeline.sacred_observer import SetID

from src.train_pipeline.train_model import train


ex = Experiment()
COLS_DF_RESULT = list(placeholder_config_competitors.create_id_dict().keys())+['metric', 'value']


@ex.config
def cfg():
    config = placeholder_config_competitors
    experiment_dir = '~/'
    experiment_root = '~/'
    seed = 0
    verbose = False


@ex.automain
def train_competitor(_run, _seed, _rnd, config: Config_Competitors, experiment_dir, experiment_root, verbose):

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

    X_test, y_test = dataset.sample(**config.sampling_kwargs, seed=seed_sampling, train=False)

    model = config.model_class(**config.model_kwargs)
    # Train and evaluate model
    result = train_comp(model = model, data_train = (X_train,y_train), data_test = (X_test,y_test), config = config, quiet = operator.not_(verbose), val_size = 0.2, _seed = _seed,
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



def simulator_competitor(config_grid: ConfigGrid_Competitors):

    ex.observers.append(FileStorageObserver(config_grid.experiment_dir))
    ex.observers.append(SetID('myid'))

    for config in config_grid.configs_from_grid():
        id = config.creat_uuid()
        ex_dir_new = os.path.join(config_grid.experiment_dir, id)
        ex.observers[1] = SetID(id)
        ex.run(config_updates={'config': config, 'experiment_dir' : ex_dir_new, 'experiment_root' : config_grid.experiment_dir,
                               'seed' : config_grid.seed, 'verbose' : config_grid.verbose
                               })

