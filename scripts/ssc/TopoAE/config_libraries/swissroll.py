import random

import numpy as np

from src.datasets.datasets import Spheres, SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae


### SWISSROLL
swissroll_midsize_euler_seed1_parallel_shuffled = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,11,num=9,base = 2.0)], len([int(i) for i in np.logspace(3,11,num=9,base = 2.0)])),
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/SwissRoll/seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(-4,4,num=9,base = 2.0)]]


swissroll_midsize_euler_seed1_parallel_shuffled_hw = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,11,num=9,base = 2.0)], len([int(i) for i in np.logspace(3,11,num=9,base = 2.0)])),
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/SwissRoll/seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(5,8,num=4,base = 2.0)]]

swissroll_midsize_lowbs_euler_seed1_parallel_shuffled_test = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,11,num=9,base = 2.0)], len([int(i) for i in np.logspace(3,11,num=9,base = 2.0)])),
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/SwissRoll/seed1_test',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(-4,4,num=9,base = 2.0)]]


swissroll_midsize_midbs_euler_seed1_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[512],
    n_epochs=[6],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,4,num=5,base = 2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/TopoAE_testing_final_3',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)



# LOCAL RUNS
swissroll_benchmark = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[j],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[0],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/benchmark',
    seed = 1,
    device = 'cpu',
    num_threads=2,
    verbose = False
) for j in [8,16,32,64,128,256,512]]


swissroll_midsize_lowbs_local_seed1_parallel_shuffled = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,11,num=9,base = 2.0)], len([int(i) for i in np.logspace(3,11,num=9,base = 2.0)])),
    n_epochs=[3],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/TopoAE_testing_final_3',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(-4,4,num=9,base = 2.0)]]


swissroll_asymmetric = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(5,9,num=5,base = 2.0)], len([int(i) for i in np.logspace(5,9,num=5,base = 2.0)])),
    n_epochs=[150],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'asymmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/asym_test3',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [i for i in np.logspace(8,9,num=2,base = 2.0)]]


swissroll_asymmetric_2 = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[1024],
    n_epochs=[150],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'asymmetric2')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/asymmetric2',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [768,1024]]


swissroll_asymmetric_push = [ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[128,256,512,768,1024],
    n_epochs=[150],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    toposig_kwargs = [dict(match_edges = 'asymmetric_push3')],
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
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/asymmetric_push4',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
) for j in [128,256,512,768,1024,1526,2024,2560]]


### TEST
swissroll_testing = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[32],
    n_epochs=[150],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs = [dict(match_edges = 'asymmetric_push3')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [3],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
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
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/test',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)