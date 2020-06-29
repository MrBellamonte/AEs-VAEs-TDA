import os
import json
import datetime
import dep.COREL_icml2019.config as config

from dep.COREL_icml2019.core.train_engine import train, configs_from_grid


if __name__ == "__main__":

    grid_one_branch = \
    {
        'train_args': {
            'learning_rate': [0.001],
            'batch_size'   : [100],
            'n_epochs'     : [50],
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [0.0, 1.0, 10.0, 20.0, 40.0]
        },
        'model_args': {
            'class_id'     : ['DCGEncDec'],
            'kwargs'       : {
                'filter_config' : [[3,32,64,128]],
                'input_config'  : [[3,32,32]],
                'latent_config' : {
                    'n_branches'         : [1],
                    'out_features_branch': [160]
                }
            }
        },
        'data_args':{
            'dataset'     : ['cifar100'],
            'subset_ratio': [1.0],
            'train'       : [True]
        }
    }

    path = '/home/simonberg/PycharmProjects/MT-VAEs-TDA'
    os.makedirs(path)

    print('start')

    train(path, grid_one_branch)
