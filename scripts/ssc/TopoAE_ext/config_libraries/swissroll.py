import numpy as np

from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE_WitnessComplex.config import ConfigGrid_TopoAE_ext
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae



swissroll_run1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[16],
    n_epochs=[15],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    match_edges = ['symmetric'],
    k = [1],
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
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/run1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for j in [int(i) for i in np.logspace(17, 20, base=2, num=4)]]



### TEST
swissroll_testing = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[16],
    n_epochs=[5],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[384],
    match_edges = ['symmetric'],
    k = [10],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 2)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/verification',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)

swissroll_testing2 = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[16],
    n_epochs=[10],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[384],
    match_edges = ['symmetric'],
    k = [10],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
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
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 2)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/verification',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)

swissroll_testing_euler = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[5],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[384],
    match_edges = ['symmetric'],
    k = [10],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/testing',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)
swissroll_testing_euler_multi = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[5],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[384],
    match_edges = ['symmetric'],
    k = [10],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 2)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/testing',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)
