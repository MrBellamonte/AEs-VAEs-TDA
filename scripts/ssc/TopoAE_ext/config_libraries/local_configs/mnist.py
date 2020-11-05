import numpy as np

from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.euler_wc_offline_configs.mnist import *
from src.datasets.datasets import SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import (
    Autoencoder_MLP_topoae, ConvAE_MNIST,
    ConvAE_MNIST_NEW, ConvAE_MNIST_SMALL)

mnist_test = ConfigGrid_WCAE(
    learning_rate=[1/100],
    batch_size=[1024],
    n_epochs=[1000],
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
    sampling_kwargs=[dict()],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=4,
        k_max=5,
        k_step=1,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.05], online_wc=[True], wc_offline = [dict(path_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2')]),
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/mnist_precomputed',
    seed=838,
    device='cpu',
    num_threads=4,
    verbose=True,
)


mnist_test2 = ConfigGrid_WCAE(
    learning_rate=[1/1000],
    batch_size=[1024],
    n_epochs=[1],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[0],
    match_edges=['push_active'],
    k=[1],
    r_max=[10],
    model_class=[ConvAE_MNIST_SMALL],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict()],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], wc_offline = [dict(path_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2')]),
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/mnist_precomputed',
    seed=838,
    device='cpu',
    num_threads=2,
    verbose=True,
)
