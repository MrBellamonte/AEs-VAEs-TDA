import numpy as np

from src.datasets.datasets import Spheres
from src.models.TopoAE.config import ConfigGrid_TopoAE
from src.models.autoencoders import Autoencoder_MLP_topoae

test_grid = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[10000],
    n_epochs=[2],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)


### All Models run on Euler


## FIRST RUNS
# regular batch sizes
eulergrid_280720 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
    n_epochs=[40],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)

# medium batch sizes
eulergrid_280720_2 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(10,12,num=3,base = 2.0)],
    n_epochs=[40],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)

# large batch sizes
eulergrid_280720_3 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[8192,10000],
    n_epochs=[40],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)