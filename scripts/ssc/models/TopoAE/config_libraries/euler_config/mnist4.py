import itertools
import random

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import DeepAE_MNIST_4D

mnist_1_deep4d = [ConfigGrid_TopoAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64, 128, 256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-5, 3, num=9, base=2.0)],
    toposig_kwargs=[dict(match_edges='symmetric')],
    model_class=[DeepAE_MNIST_4D],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=False,
        save_train_latent=False,
        online_visualization=False,
        quant_eval=True,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid=[''],
    method_args=[None],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_topoae_1_deepae4',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False
) for seed in [838,579,1988]]


mnist_1_deep4d_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1_deep4d]))