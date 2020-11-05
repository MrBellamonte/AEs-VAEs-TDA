import itertools

import numpy as np

from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_MNIST

wcpath_mnist_s838_256='/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs256-seed838-noiseNone-4a5487de'
wcpath_mnist_s838_512='/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs512-seed838-noiseNone-ced06774'
wcpath_mnist_s838_1024='/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2'


mnist_s838_256_1 = ConfigGrid_WCAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2],
    r_max=[10],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125], online_wc=[True], wc_offline = [dict(path_to_data = wcpath_mnist_s838_256)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)


mnist_s838_512_1 = ConfigGrid_WCAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2],
    r_max=[10],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125], online_wc=[True], wc_offline = [dict(path_to_data = wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False,
)

mnist_s838_1024_1 = ConfigGrid_WCAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2],
    r_max=[10],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125], online_wc=[True], wc_offline = [dict(path_to_data = wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False,
)


mnist_s838_1 = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in [mnist_s838_256_1, mnist_s838_512_1,mnist_s838_1024_1]]))


mnist_s838_1024_cuda_test = ConfigGrid_WCAE(
    learning_rate=[1/1000],
    batch_size=[1024],
    n_epochs=[5],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    match_edges=['push_active'],
    k=[1],
    r_max=[10],
    model_class=[ConvAE_MNIST],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], wc_offline = [dict(path_to_data = wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=True,
)
