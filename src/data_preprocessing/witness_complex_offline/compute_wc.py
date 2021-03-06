"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver

from src.data_preprocessing.witness_complex_offline.config import (
    ConfigWC, placeholder_config_wc,
    ConfigWC_Grid)
from src.data_preprocessing.witness_complex_offline.wc_offline_utils import compute_wc_offline

from src.train_pipeline.sacred_observer import SetID

from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'
ex = Experiment()



@ex.config
def cfg():
    config = placeholder_config_wc
    seed = 0

@ex.automain
def run(_run, _seed, _rnd, config: ConfigWC):
    compute_wc_offline(config)

def compute_wc_multiple(config_grid):
    ex.observers.append(FileStorageObserver(config_grid.root_path))
    ex.observers.append(SetID('myid'))
    for config in config_grid.configs_from_grid():
        ex.observers[1] = SetID(config.uid)
        ex.run(config_updates={'config': config, 'seed': config.seed})










