import datetime
import os

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from src.datasets.datasets import create_sphere_dataset, Spheres
from src.model.autoencoders import autoencoder
from src.model.loss_collection import L1Loss
from src.model.COREL.train_engine import (
    train_new)
from src.utils.config_utils import configs_from_grid


if __name__ == "__main__":

    # config_grid = {
    #     'train_args': {
    #         'learning_rate': [0.001],
    #         'batch_size'   : [32,64,128,256,512],
    #         'n_epochs'     : [50],
    #         'rec_loss_w'   : [1.0],
    #         'top_loss_w'   : [1/4,1/8,1/16,1/32,1/64,1/128,1/256, 1/512, 1/1024,0],
    #     },
    #     'model_args': {
    #         'class_id': ['autoencoder'],
    #         'kwargs'  : {
    #             'input_dim'         : [101],
    #             'latent_dim'        : [2],
    #             'size_hidden_layers': [[128,64,32]]
    #         }
    #     }
    # }

    # config_grid = {
    #     'train_args': {
    #         'learning_rate': [0.001],
    #         'batch_size'   : [32,64,128,256,512],
    #         'n_epochs'     : [30],
    #         'rec_loss_w'   : [1.0],
    #         'top_loss_w'   : [8,4,2,1,1/2,1/4,1/8,1/16,1/32,1/64,1/128],
    #     },
    #     'model_args': {
    #         'class_id': ['autoencoder'],
    #         'kwargs'  : {
    #             'input_dim'         : [101],
    #             'latent_dim'        : [2],
    #             'size_hidden_layers': [[128,64,32]]
    #         }
    #     }
    # }

    # config_grid = {
    #     'train_args': {
    #         'learning_rate': [0.001],
    #         'batch_size'   : [32,64,128,256, 512],
    #         'n_epochs'     : [50],
    #         'rec_loss_w'   : [1.0],
    #         'top_loss_w'   : [1/1024, 1/512, 1/256, 1/128, 1/64, 1/32,1/16, 1/8, 1/4, 1,1/2, 2, 4, 8, 16, 32],
    #     },
    #     'model_args': {
    #         'class_id': ['autoencoder'],
    #         'kwargs'  : {
    #             'input_dim'         : [101],
    #             'latent_dim'        : [2],
    #             'size_hidden_layers': [[128,64,32]]
    #         }
    #     }
    #}

    config_grid = {
        'train_args': {
            'learning_rate': [0.001],
            'batch_size'   : [32,64,128,256, 512],
            'n_epochs'     : [50],
            'rec_loss' : {
                'loss_class' : [L1Loss()],
                'weight' : [1]
            },
            'top_loss': {
                'loss_class': [L1Loss()],
                'weight'    : [1]
            },
        },
        'model_args': {
            'model_class': [autoencoder],
            'kwargs'  : {
                'input_dim'         : [101],
                'latent_dim'        : [2],
                'size_hidden_layers': [[128,64,32]]
            }
        },
        'data_args':{
            'dataset' : Spheres(),
            'kwargs' :{
                'n_samples' : 500
            }
        }
    }

    X, y = create_sphere_dataset()


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/TEST'
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
        #train(dataset, config, path)
        #train_Huber10(dataset, config, path)
        #train_inctlw(dataset, config, path)
        train_new(dataset, config, path)
