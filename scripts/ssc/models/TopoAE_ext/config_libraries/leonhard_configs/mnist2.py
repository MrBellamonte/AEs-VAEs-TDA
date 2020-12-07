import itertools

import numpy as np

from scripts.ssc.models.TopoAE_ext.config_libraries.euler_configs.mnist import (
    wcpath_mnist_s838_1024,
    wcpath_mnist_s838_512, wcpath_mnist_s838_256, wcpath_mnist_s838_64, wcpath_mnist_s838_128)
from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_MNIST, DeepAE_MNIST

mnist_1_k12_deepae = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[bs],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-8, 0, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2],
    r_max=[10],
    model_class=[DeepAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=path_to_data)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist_wae_1_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for bs,path_to_data in [(64,wcpath_mnist_s838_64),(128,wcpath_mnist_s838_128),(256,wcpath_mnist_s838_256),(512,wcpath_mnist_s838_512),(1024,wcpath_mnist_s838_1024)]]


mnist_1_k12_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1_k12_deepae]))

mnist_1_k48_deepae = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[bs],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-8, 0, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[4,8],
    r_max=[10],
    model_class=[DeepAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=path_to_data)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist_wae_1_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for bs,path_to_data in [(64,wcpath_mnist_s838_64),(128,wcpath_mnist_s838_128),(256,wcpath_mnist_s838_256),(512,wcpath_mnist_s838_512),(1024,wcpath_mnist_s838_1024)]]

mnist_1_k48_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1_k48_deepae]))


mnist_1_k1216_deepae = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[bs],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-8, 0, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[12,16],
    r_max=[10],
    model_class=[DeepAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=path_to_data)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist_wae_1_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for bs,path_to_data in [(64,wcpath_mnist_s838_64),(128,wcpath_mnist_s838_128),(256,wcpath_mnist_s838_256),(512,wcpath_mnist_s838_512),(1024,wcpath_mnist_s838_1024)]]

mnist_1_k1216_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1_k1216_deepae]))



