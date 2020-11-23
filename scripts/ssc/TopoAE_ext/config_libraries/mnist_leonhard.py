import numpy as np

from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.mnist import (
    wcpath_mnist_s838_1024,
    wcpath_mnist_s838_512, wcpath_mnist_s838_256)
from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_MNIST




mnist_s838_1024_lw = ConfigGrid_WCAE(
    learning_rate=[1/10,1/100, 1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(-2, -1, num=2, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2, 3, 4, 5, 6],
    r_max=[10],
    model_class=[ConvAE_MNIST],
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
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024_2',
    seed=838,
    device='cuda',
    num_threads=4,
    verbose=False,
)


mnist_s838_512_lw = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(-2, -1, num=2, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2, 3, 4, 5, 6],
    r_max=[10],
    model_class=[ConvAE_MNIST],
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
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512_2',
    seed=838,
    device='cuda',
    num_threads=4,
    verbose=False,
)



mnist_s838_256_lw = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(-2, -1, num=2, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2, 3, 4, 5, 6],
    r_max=[10],
    model_class=[ConvAE_MNIST],
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
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_256)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist256_2',
    seed=838,
    device='cuda',
    num_threads=4,
    verbose=False)