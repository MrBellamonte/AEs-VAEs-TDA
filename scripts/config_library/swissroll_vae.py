import itertools
import random

import numpy as np

from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.wc_offline_configs.swissroll import \
    (
    SWISSROLL_NONOISE36, SWISSROLL_NONOISE36_LOCALSERVER)
from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.variational_autoencoder.varautoencoders import VanillaVAE


vae_test = ConfigGrid_WCAE(
    learning_rate=[1/100],
    batch_size=[128],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[3],
    r_max=[10],
    model_class=[VanillaVAE],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [640] #2560
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=True,
        quant_eval=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1.25], online_wc=[True], wc_offline=[None]),
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/vae_test',
    seed=5310,
    device='cpu',
    num_threads=1,
    verbose=True,
)



bs = [64,128,256,512]


vae_run1_seed36 = [ConfigGrid_WCAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[bs],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[1,2,4],
    r_max=[10],
    model_class=[VanillaVAE],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]],
        'lambda_kld': list(np.logspace(-3,1,5))
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=True,
        quant_eval=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.05,1.1,1.15,1.2,1.25], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/vae_test',
    seed=36,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip([36]*len(bs), bs, SWISSROLL_NONOISE36)]

vae_run1_seed36 = [ConfigGrid_WCAE(
    learning_rate=[1/10],
    batch_size=[bs],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[8192],
    match_edges=['push_active'],
    k=[1],
    r_max=[10],
    model_class=[VanillaVAE],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]],
        'lambda_kld': list(np.logspace(-3,1,5))
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        quant_eval=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/vae_test',
    seed=36,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip([36]*len(bs), bs, SWISSROLL_NONOISE36)]


vae_run1_seed36_remote = [ConfigGrid_WCAE(
    learning_rate=[1/10,1/100],
    batch_size=[bs],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(10,13,num=4,base = 2.0)],
    match_edges=['push_active'],
    k=[1,2,4],
    r_max=[10],
    model_class=[VanillaVAE],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]],
        'lambda_kld': list(np.logspace(-3,1,5))
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        quant_eval=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125,1.25], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA/output/WAE/vae_run1',
    seed=36,
    device='cpu',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip([36]*len(bs), bs, SWISSROLL_NONOISE36_LOCALSERVER)]


vae_run1_seed36_remote_cuda = [ConfigGrid_WCAE(
    learning_rate=[1/10,1/100],
    batch_size=[bs],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(10,13,num=4,base = 2.0)],
    match_edges=['push_active'],
    k=[1,2,4],
    r_max=[10],
    model_class=[VanillaVAE],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]],
        'lambda_kld': list(np.logspace(-3,1,5))
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=True,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        quant_eval=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125,1.25], online_wc=[True], wc_offline = [dict_wc]),
    experiment_dir='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA/output/WAE/vae_run1',
    seed=36,
    device='cuda',
    num_threads=1,
    verbose=False,
) for seed, bs, dict_wc in zip([36]*len(bs), bs, SWISSROLL_NONOISE36_LOCALSERVER)]


vae_run1_seed36_list_remote = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in vae_run1_seed36_remote]))
vae_run1_seed36_list_remote_cuda = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in vae_run1_seed36_remote_cuda]))
vae_run1_seed36_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in vae_run1_seed36]))
