import itertools

from src.datasets.datasets import MNIST_offline, Unity_XYTransOpenAI
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import (
    DeepAE_MNIST, DeepAE_MNIST_3D, DeepAE_MNIST_4D,
    DeepAE_MNIST_8D, ConvAE_Unity480320)
from src.models.vanillaAE.config import ConfigGrid_VanillaAE


xy_trans_final_notopo10= ConfigGrid_VanillaAE(
    learning_rate=[1/10],
    batch_size=[200],
    n_epochs=[2000],
    weight_decay=[1e-6],
    early_stopping=[200],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=10,
        k_step=5,
        quant_eval=False

    )],
    uid=[''],
    method_args=dict(val_size=[0]),
    experiment_dir='/cluster/scratch/schsimo/output/xy_trans_final_notopo',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False)

xy_trans_final_notopo100 = ConfigGrid_VanillaAE(
    learning_rate=[1/100],
    batch_size=[200],
    n_epochs=[2000],
    weight_decay=[1e-6],
    early_stopping=[200],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=10,
        k_step=5,
        quant_eval=False

    )],
    uid=[''],
    method_args=dict(val_size=[0]),
    experiment_dir='/cluster/scratch/schsimo/output/xy_trans_final_notopo',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False)

xy_trans_final_notopo1000 = ConfigGrid_VanillaAE(
    learning_rate=[1/1000],
    batch_size=[200],
    n_epochs=[2000],
    weight_decay=[1e-6],
    early_stopping=[200],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=10,
        k_step=5,
        quant_eval=False

    )],
    uid=[''],
    method_args=dict(val_size=[0]),
    experiment_dir='/cluster/scratch/schsimo/output/xy_trans_final_notopo',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

