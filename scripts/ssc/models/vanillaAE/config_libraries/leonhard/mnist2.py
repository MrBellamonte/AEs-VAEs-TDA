import itertools

from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import (
    DeepAE_MNIST, DeepAE_MNIST_3D, DeepAE_MNIST_4D,
    DeepAE_MNIST_8D)
from src.models.vanillaAE.config import ConfigGrid_VanillaAE




mnist_2_deepae_2 = [ConfigGrid_VanillaAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64,128,256,512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
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
    method_args=[dict()],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_2_notopo_deepae_2',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for seed in [838,579,1988]]

mnist_2_deepae_2_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_2_deepae_2]))

mnist_3_deepae = [ConfigGrid_VanillaAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64,128,256,512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    model_class=[DeepAE_MNIST_3D],
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
    method_args=[dict()],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_3_notopo_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for seed in [838,579,1988]]

mnist_3_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_3_deepae]))

mnist_4_deepae = [ConfigGrid_VanillaAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64,128,256,512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    model_class=[DeepAE_MNIST_4D],
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
    method_args=[dict()],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_4_notopo_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for seed in [838,579,1988]]


mnist_4_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_4_deepae]))

mnist_8_deepae = [ConfigGrid_VanillaAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64,128,256,512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    model_class=[DeepAE_MNIST_8D],
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
    method_args=[dict()],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_8_notopo_deepae',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False) for seed in [838,579,1988]]

mnist_8_deepae_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_8_deepae]))
