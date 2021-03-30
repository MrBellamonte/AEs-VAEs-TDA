import random

import numpy as np

from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae

# SYMMETRIC EDGE MATCHING

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (100 configs per grid), 15 processes
symmetric_grid1 = [ConfigGrid_WCAE(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['symmetric'],
    k=[1, 2, 4, 8, 16],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_symmetric',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for lr, seed in zip(list(np.repeat([1/10, 1/100, 1/1000], 5)), [1452, 1189, 1573, 959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (100 configs per grid), 15 processes
symmetric_grid2 = [ConfigGrid_WCAE(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['symmetric'],
    k=[1, 2, 4, 8, 16],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_symmetric',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for lr, seed in zip(list(np.repeat([1/10, 1/100, 1/1000], 5)), [3859, 2525, 2068, 3302, 2517]*3)]

# ASYMMETRIC EDGE MATCHING (PUSH)

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (100 configs per grid), 15 processes
asym_grid1 = [ConfigGrid_WCAE(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push'],
    k=[1, 2, 4, 8, 16],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_push',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for lr, seed in zip(list(np.repeat([1/10, 1/100, 1/1000], 5)), [1452, 1189, 1573, 959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (100 configs per grid), 15 processes
asym_grid2 = [ConfigGrid_WCAE(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push'],
    k=[1, 2, 4, 8, 16],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_push',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for lr, seed in zip(list(np.repeat([1/10, 1/100, 1/1000], 5)), [3859, 2525, 2068, 3302, 2517]*3)]

# ASYMMETRIC EDGE MATCHING (ACTIVE PUSH)

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (300 configs per grid), 15 processes
asymapush_grid1 = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2, 4, 8],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=mu_push)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for mu_push, seed in zip(list(np.repeat([1.05, 1.1, 1.15], 5)), [1452, 1189, 1573, 959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (300 configs per grid!), 15 processes
asymapush_grid2 = [ConfigGrid_WCAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(6, 9, num=4, base=2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2, 4, 8],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=mu_push)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for mu_push, seed in zip(list(np.repeat([1.05, 1.1, 1.15], 5)), [3859, 2525, 2068, 3302, 2517]*3)]

# CHECK STABILITY OF METHOD
asymapush_stability = [ConfigGrid_WCAE(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-7],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560],
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush_stability',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed in [91, 561, 4]]

asymapush_stability2 = [ConfigGrid_WCAE(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-7],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560],
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush_stability',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed in [647, 8, 36]]

asymapush_stability3 = [ConfigGrid_WCAE(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-7],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560],
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush_stability',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed in [790, 933, 270]]

asymapush_stability4 = [ConfigGrid_WCAE(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-7],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560],
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush_stability',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed in [867, 588, 146]]



# asym_push bs64
asymapush_grid_bs64 = [ConfigGrid_WCAE(
    learning_rate=[lr],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9, 13, num=5, base=2.0)],
    match_edges=['push_active'],
    k=[1, 2],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25),dict(n_jobs=1, normalize=True, mu_push=1.35)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False,
) for lr, seed in zip(list(np.repeat([1/10, 1/100, 1/1000], 10)), [1452, 1189, 1573,  959, 1946,3859, 2525, 2068, 3302, 2517]*3)]


# RERUN "FINALISTS" FOR EVAL
wctopoae_finalists = [ConfigGrid_WCAE(
    learning_rate=[1/1000],
    batch_size=[256],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[2048],
    match_edges=['push_active'],
    k=[1],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.05)],
    experiment_dir='/cluster/scratch/schsimo/output/finalists',
    seed=3302,
    device='cpu',
    num_threads=1,
    verbose=False,
),
ConfigGrid_WCAE(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[2],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/finalists',
    seed=561,
    device='cpu',
    num_threads=1,
    verbose=False,
),
ConfigGrid_WCAE(
    learning_rate=[1/100],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[512],
    match_edges=['push_active'],
    k=[1],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs=1, normalize=True, mu_push=1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/finalists',
    seed=2517,
    device='cpu',
    num_threads=1,
    verbose=False,
)
]