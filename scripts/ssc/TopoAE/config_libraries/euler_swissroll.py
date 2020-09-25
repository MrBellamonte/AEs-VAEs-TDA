import random

import numpy as np

from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae



swissroll_multiseed_final = [ConfigGrid_TopoAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid = [''],
    method_args = [None],
    experiment_dir='/cluster/scratch/schsimo/output/TopoAE_swissroll_symmetric',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for seed in [1452, 1189, 1573, 959, 1946,3859, 2525, 2068, 3302, 2517]]


