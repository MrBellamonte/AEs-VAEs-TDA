import random

import numpy as np

from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE_WitnessComplex.config import ConfigGrid_TopoAE_ext
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae


# SYMMETRIC EDGE MATCHING

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (100 configs per grid), 15 processes
symmetric_grid1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['symmetric'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_symmetric',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for lr, seed in zip(list(np.repeat([1/10,1/100,1/1000],5)),[1452, 1189, 1573,  959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (100 configs per grid), 15 processes
symmetric_grid2 = [ConfigGrid_TopoAE_ext(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['symmetric'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_symmetric',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for lr, seed in zip(list(np.repeat([1/10,1/100,1/1000],5)),[3859, 2525, 2068, 3302, 2517]*3)]

# ASYMMETRIC EDGE MATCHING (PUSH)

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (100 configs per grid), 15 processes
asym_grid1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['push'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_push',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for lr, seed in zip(list(np.repeat([1/10,1/100,1/1000],5)),[1452, 1189, 1573,  959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (100 configs per grid), 15 processes
asym_grid2 = [ConfigGrid_TopoAE_ext(
    learning_rate=[lr],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['push'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_push',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for lr, seed in zip(list(np.repeat([1/10,1/100,1/1000],5)),[3859, 2525, 2068, 3302, 2517]*3)]

# ASYMMETRIC EDGE MATCHING (ACTIVE PUSH)

# grid 1 -> seeds = [1452, 1189, 1573,  959, 1946] (300 configs per grid), 15 processes
asymapush_grid1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['push'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True, mu_push = mu_push)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for mu_push, seed in zip(list(np.repeat([1.05,1.1,1.15],5)),[1452, 1189, 1573,  959, 1946]*3)]

# grid 2 -> seeds = [3859, 2525, 2068, 3302, 2517] (300 configs per grid!), 15 processes
asymapush_grid2 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(6,9,num=4,base = 2.0)], 4),
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(9,13,num=5,base = 2.0)],
    match_edges = ['push'],
    k = [1,2,4,8,16],
    r_max = [10],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True, mu_push = mu_push)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for mu_push, seed in zip(list(np.repeat([1.05,1.1,1.15],5)),[3859, 2525, 2068, 3302, 2517]*3)]


# CHECK STABILITY OF METHOD

asymapush_stability = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/25],
    batch_size=[128],
    n_epochs=[1000],
    weight_decay=[1e-7],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges = ['push_active'],
    k = [2],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560],
        'seed' : [32, 539, 934, 331, 517]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 20,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1, normalize = True, mu_push = 1.25)],
    experiment_dir='/cluster/scratch/schsimo/output/WCTopoAE_swissroll_apush_stability',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for seed in [574,  85, 136]]