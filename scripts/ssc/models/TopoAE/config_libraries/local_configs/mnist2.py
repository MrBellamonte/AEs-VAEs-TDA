import random

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae, ConvAE_MNIST

mnist_test_loc = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[64,],
    n_epochs=[1],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict()],
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = False,
        online_visualization = False,
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    method_args = [None],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/MNIST/test',
    seed = 838,
    device = 'cpu',
    num_threads=1,
    verbose = True
)


