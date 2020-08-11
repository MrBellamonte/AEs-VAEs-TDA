import random

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae


### SWISSROLL
spheres_euler_seed6_parallel_shuffled = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,12,num=10,base = 2.0)], 10),
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [11],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [512]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/Spheres/seed6',
    seed = 6,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(-4,8,num=13,base = 2.0)]]