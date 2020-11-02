import itertools

import numpy as np

from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.euler_wc_offline_configs.swissroll_nonoise import \
    (
    SWISSROLL_NONOISE3288, SWISSROLL_NONOISE_all, SWISSROLL_NONOISE_h1, SWISSROLL_NONOISE_h2)
from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae

seeds = [36, 3851, 2570, 4304, 1935, 7954, 5095, 5310, 1577, 3288]
seeds_h1 = [36, 3851, 2570, 4304, 1935]
seeds_h2 = [7954, 5095, 5310, 1577, 3288]
bs = [64,128,256,512]

bs_all = len(seeds)*bs
bs_all_h1 = len(seeds_h1)*bs
bs_all_h2 = len(seeds_h2)*bs

seeds_all = np.repeat(seeds,4)
seeds_h1_all = np.repeat(seeds_h1,4)
seeds_h2_all = np.repeat(seeds_h1,4)

swissroll_h1 = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[int(bs)],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2,3,4,5,6],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=15,
        k_max=45,
        k_step=15,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.05,1.1,1.15,1.2,1.25], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/cluster/scratch/schsimo/output/WCAE_swissroll_nonoise',
    seed=int(seed),
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip(seeds_h1_all, bs_all_h1, SWISSROLL_NONOISE_h1)]

# swissroll_h2 = [ConfigGrid_WCAE(
#     learning_rate=[1/10, 1/100, 1/1000],
#     batch_size=[int(bs)],
#     n_epochs=[1000],
#     weight_decay=[1e-6],
#     early_stopping=[50],
#     rec_loss_weight=[1],
#     top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
#     match_edges=['push_active'],
#     k=[1,2,3,4,5,6],
#     r_max=[10],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[32, 32]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [2560]
#     },
#     eval=[ConfigEval(
#         active=True,
#         evaluate_on='test',
#         eval_manifold=True,
#         save_eval_latent=True,
#         save_train_latent=True,
#         online_visualization=False,
#         k_min=15,
#         k_max=45,
#         k_step=15,
#     )],
#     uid=[''],
#     toposig_kwargs=[dict()],
#     method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.05,1.1,1.15,1.2,1.25], online_wc=[True], wc_offline = [dict_wc]),
#     experiment_dir='/cluster/scratch/schsimo/output/WCAE_swissroll_nonoise',
#     seed=int(seed),
#     device='cpu',
#     num_threads=1,
#     verbose=False,
# ) for seed, bs, dict_wc in zip(seeds_h2_all, bs_all_h2, SWISSROLL_NONOISE_h2)]


swissroll_h22 = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[int(bs)],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1,2,3,4,5,6],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=15,
        k_max=45,
        k_step=15,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.05,1.1,1.15,1.2,1.25], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/cluster/scratch/schsimo/output/WCAE_swissroll_nonoise2',
    seed=int(seed),
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip(seeds_h2_all, bs_all_h2, SWISSROLL_NONOISE_h2)]

swissroll_h22_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in swissroll_h22]))