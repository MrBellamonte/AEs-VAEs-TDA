import itertools
import random

import numpy as np

from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_MNIST

wcpath_mnist_s838_64 = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs64-seed838-noiseNone-20738678'
wcpath_mnist_s838_128 = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs128-seed838-noiseNone-4f608157'
wcpath_mnist_s838_256 = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs256-seed838-noiseNone-4a5487de'
wcpath_mnist_s838_512 = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs512-seed838-noiseNone-ced06774'
wcpath_mnist_s838_1024 = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2'

# 10h per model, 360 models, -> 40 processes, approx 90h -> 30h margin
mnist_s838_64_fullk_fullnu = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_64_hihik_fullnu = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[8,10,12,14],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_64_fullk_fullnu_hldb = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(5, 9, num=5, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)],
                     ),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)



mnist_s838_64_fullk_fullnu_lldb = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-2, -1, num=2, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)],
                     ),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64_2',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_64_fullk_fullnu_llldb = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4, -3, num=2, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)],
                     ),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64_3',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)


mnist_s838_64_fullk_fullnu_hldb_decay1 = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[1],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_64)],
                     lam_t_decay=[
                         {0  : 4098, 50: 2048, 100: 1024, 150: 512, 200: 256, 250: 128, 300: 64,
                          350: 32,
                          400: 16, 450: 8, 500: 4},
                         {0  : 8192, 50: 4098, 100: 2048, 150: 1024, 200: 512, 250: 256, 300: 128,
                          350: 64,
                          400: 32, 450: 16, 500: 8},
                         {0  : 2048, 50: 1024, 100: 512, 150: 256, 200: 128, 250: 64, 300: 32,
                          350: 16,
                          400: 8, 450: 4, 500: 2}]
                     ),
    experiment_dir='/cluster/scratch/schsimo/output/mnist64',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)


mnist_s838_128_fullk_fullnu = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_128)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist128',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_128_fullk_fullnu_hlbd = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(5, 9, num=5, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_128)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist128',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)





mnist_s838_128_fullk_fullnu_llbd = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-2, -1, num=2, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_128)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist128_2',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_128_fullk_fullnu_lllbd = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4, -3, num=2, base=2.0)],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_128)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist128_3',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)


mnist_s838_128_fullk_fullnu_decay1 = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[1],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25, 1.375],
                     online_wc=[True], wc_offline=[dict(path_to_data=wcpath_mnist_s838_128)],
                     lam_t_decay=[
                         {0  : 4098, 50: 2048, 100: 1024, 150: 512, 200: 256, 250: 128, 300: 64,
                          350: 32,
                          400: 16, 450: 8, 500: 4},
                         {0  : 8192, 50: 4098, 100: 2048, 150: 1024, 200: 512, 250: 256, 300: 128,
                          350: 64,
                          400: 32, 450: 16, 500: 8},
                         {0  : 2048, 50: 1024, 100: 512, 150: 256, 200: 128, 250: 64, 300: 32,
                          350: 16,
                          400: 8, 450: 4, 500: 2}
                     ]
                     ),
    experiment_dir='/cluster/scratch/schsimo/output/mnist128',
    seed=838,
    device='cpu',
    num_threads=1,
    verbose=False)

mnist_s838_256_1 = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
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
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_256_hinu = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_256)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_256_hinu_mik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4, 5, 6],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_256)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_256_hinu_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[5, 6],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_256)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_256_mik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4],
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
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_256_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[5, 6],
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
    experiment_dir='/cluster/scratch/schsimo/output/mnist256',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False)

mnist_s838_512_1 = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
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
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)
mnist_s838_512_mik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4],
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
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_512_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[5, 6],
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
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_512_hinu_mik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4],
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
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_512_hinu_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[5, 6],
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
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_512)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist512',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_1024_1 = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
    r_max=[10],
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
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_1024_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4, 5, 6],
    r_max=[10],
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
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_1024_hinu = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
    r_max=[10],
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
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)

mnist_s838_1024_hinu_hik = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[1024],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 4, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[3, 4, 5, 6],
    r_max=[10],
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
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25, 1.375], online_wc=[True],
                     wc_offline=[dict(path_to_data=wcpath_mnist_s838_1024)]),
    experiment_dir='/cluster/scratch/schsimo/output/mnist1024',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=False,
)
