from src.datasets.datasets import Spheres
from src.model.autoencoders import autoencoder
from src.model.loss_collection import L1Loss


config_grid_testSpheres = {
    'train_args': {
        'learning_rate': [1/1000],
        'batch_size'   : [256, 512],
        'n_epochs'     : [2],
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
            'size_hidden_layers': [[128 ,64 ,32]]
        }
    },
    'data_args' :{
        'dataset' : Spheres(),
        'kwargs' :{
            'n_samples': 500
        }
    }
}