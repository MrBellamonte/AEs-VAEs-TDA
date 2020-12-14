import itertools

import numpy as np

from src.competitors.competitor_models import UMAP, tSNE
from src.competitors.config import ConfigGrid_Competitors
from src.datasets.datasets import SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval


umap_mnist_euler_multi = [ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=dict(test_eval=[False],n_neighbors = np.linspace(10,500,15).astype(int).tolist(), min_dist = np.linspace(0.1,1,10).tolist()),
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA', n_samples = 10000)],
    eval=[ConfigEval(
        active=True,
        evaluate_on=None,
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid = [''],
    experiment_dir='/cluster/scratch/schsimo/output/umap_mnist2',
    seed = seed,
    verbose = True
) for seed in [1110,1909,1318]]

umap_mnist_euler_multi_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in umap_mnist_euler_multi]))

umap_mnist_test_local = ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=dict(test_eval=[False],n_neighbors = [70], min_dist = [0.05]),
    dataset=[MNIST_offline()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on=None,
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid = [''],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/competitors/mnist_testing',
    seed = 1,
    verbose = True
)




tsne_mnist_euler_multi = [ConfigGrid_Competitors(
    model_class = [tSNE],
    model_kwargs=dict(n_jobs = [1],perplexity = np.linspace(5,50,5).astype(int).tolist()),
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA', n_samples = 10000)],
    eval=[ConfigEval(
        active=True,
        evaluate_on=None,
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid = [''],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_tsne',
    seed = seed,
    verbose = True
) for seed in [1110,1909,1318]]

umap_mnist_euler_multi_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in umap_mnist_euler_multi]))


