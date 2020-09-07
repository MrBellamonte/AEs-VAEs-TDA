import os

import pandas as pd
import torch
from sacred import Experiment
from torch.utils.data import TensorDataset

from src.models.MLDL.MLDL_AE import MLDL_model
from src.models.MLDL.config import ConfigMLDL
from src.models.MLDL.loss import MLDL_Loss

#hardcoded for the time-being
DEVICE = 'cpu'

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

def SetModel(param):
    Model = MLDL_model(param).to(DEVICE)
    loss = MLDL_Loss(args=param, cuda=DEVICE)

    return Model, loss

@ex.automain
def train_MLDL(_run, _seed, _rnd, config: ConfigMLDL, experiment_dir, experiment_root, device, num_threads, verbose):
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