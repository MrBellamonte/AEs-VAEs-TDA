import itertools
import random

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae, ConvAE_MNIST

mnist_test = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs=[dict(match_edges='symmetric')],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        k_min=10,
        k_max=30,
        k_step=5,
    )],
    uid=[''],
    method_args=[None],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_topoae_test',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False
)

mnist_1 = [ConfigGrid_TopoAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64, 128, 256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-5, 3, num=9, base=2.0)],
    toposig_kwargs=[dict(match_edges='symmetric')],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        quant_eval=True,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid=[''],
    method_args=[None],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_topoae_1',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False
) for seed in [838,579,1988,1474]]

mnist_1_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1]))
