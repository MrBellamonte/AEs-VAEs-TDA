import datetime
import os

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from src.datasets.datasets import double_tours
from src.model.COREL.train_engine import train, configs_from_grid

if __name__ == "__main__":

    config_grid = {
        'train_args': {
            'learning_rate': [0.001],
            'batch_size'   : [32,64,128],
            'n_epochs'     : [25],
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [1/32,1/16,1/8,1/4],
        },
        'model_args': {
            'class_id': ['autoencoder'],
            'kwargs'  : {
                'input_dim'         : [3],
                'latent_dim'        : [2],
                'size_hidden_layers': [[16,8]]
            }
        }
    }



    X, y = double_tours(n_samples=1000)


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/double_torus_test'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    configs = configs_from_grid(config_grid)
    for i, config in enumerate(configs):
        print('Run models for configuration {} out of {}'.format(i+1, len(configs)))
        torch.cuda.empty_cache()
        train(dataset, config, path)
