from fractions import Fraction

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll
from src.models.TopoAE.config import ConfigGrid_TopoAE
from src.models.autoencoders import Autoencoder_MLP_topoae

test_grid = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[512],
    n_epochs=[2],
    weight_decay=[0],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
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

# Spheres

moor_config_approx_1 = ConfigGrid_TopoAE(
    learning_rate=[27/100000],
    batch_size=[28],
    n_epochs=[100],
    weight_decay=[1e-05],
    rec_loss_weight=[1],
    top_loss_weight=[float(Fraction(22/51))],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)


# Swiss Roll
swiss_roll_nonoise_benchmark_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size= [int(i) for i in np.logspace(2,10,num=9,base = 2.0)] + [1536],
    n_epochs=[40],
    weight_decay=[None],
    rec_loss_weight=[1],
    top_loss_weight=[0],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)



swiss_roll_nonoise_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[2, 4],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)

swiss_roll_nonoise_2 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/2, 1],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)

swiss_roll_nonoise_3 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/8, 1/4],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)

swiss_roll_nonoise_4 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/32, 1/16],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)


swiss_roll_nonoise_5 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size= [int(i) for i in np.logspace(2,9,num=9,base = 2.0)] + [1536],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[8,16],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[16, 8]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [1536]
    }
)

### All Models run on Euler


## FIRST RUNS
# regular batch sizes
eulergrid_280720 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
    n_epochs=[40],
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
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
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
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
    weight_decay = [None],
    rec_loss_weight=[1],
    top_loss_weight=[1/4,1/2,1,2,4],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
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