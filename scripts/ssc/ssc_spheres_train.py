import datetime
import os

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.datasets import create_sphere_dataset
from src.datasets.shapes import dsphere
from src.model.train_engine import train, configs_from_grid

if __name__ == "__main__":

    config_grid = {
        'train_args': {
            'learning_rate': [0.001],
            'batch_size'   : [256],
            'n_epochs'     : [50],
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [1/32,1/16,2/16,3/16],
        },
        'model_args': {
            'class_id': ['autoencoder'],
            'kwargs'  : {
                'input_dim'         : [101],
                'latent_dim'        : [2],
                'size_hidden_layers': [[128, 64, 32]]
            }
        }
    }

    X, y = create_sphere_dataset()


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    configs = configs_from_grid(config_grid)
    for i, config in enumerate(configs):
        print('Run model for configuration {} out of {}'.format(i+1, len(configs)))
        train(dataset, config, path)
