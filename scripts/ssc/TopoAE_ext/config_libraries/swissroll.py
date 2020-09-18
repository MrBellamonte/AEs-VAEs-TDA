import random

import numpy as np

from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE_WitnessComplex.config import ConfigGrid_TopoAE_ext
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae

### SWISSROLL MULTISEED
k1_multiseed = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 15,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed_new',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)  for tlw, seed in zip(list(np.repeat([i for i in np.logspace(9,13,num=5,base = 2.0)],9)),[8,12,31,39,91,102,104,309,567]*5)]

k1_multiseed2 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 15,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed_new',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)  for tlw, seed in zip(list(np.repeat([i for i in np.logspace(9,13,num=5,base = 2.0)],9)),[643,666,678,789,809,1000,1094,1333,1600]*5)]

k1_multiseed3 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 15,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed_new',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)  for tlw, seed in zip(list(np.repeat([i for i in np.logspace(9,13,num=5,base = 2.0)],9)),[2643,2666,2678,2789,2809,3000,3094,3333,3600]*5)]

k1_multiseed4 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 15,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed_new',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)  for tlw, seed in zip(list(np.repeat([i for i in np.logspace(9,13,num=5,base = 2.0)],9)),[3643,3666,3678,3789,3809,4000,4094,4333,4600]*5)]



### SWISSROLL KN-MULTISEED

k1_multiseed = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[30],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
    match_edges = ['symmetric'],
    k = [2,4,8],
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
        k_max = 15,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/kn_multiseed_new',
    seed = seed,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)  for tlw, seed in zip(list(np.repeat([i for i in np.logspace(9,13,num=5,base = 2.0)],9)),[6019,6023,6187,6199,6203,6205,6207,6213,6271]*5)]

### SWISSROLL SEED 102
k1seed102 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[15],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        'n_samples': [2560] #2560
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1seed102',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw in [int(i) for i in np.logspace(1, 13, base=2, num=13)]]


### SWISSROLL - MULTISEED K1 ONLY (DO NOT CHANGE!)
swissroll_k1multiseed_parallel_batch1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw, seed in zip(list(np.repeat([i for i in np.logspace(1,12,num=12,base = 2.0)],4)),[6,34,79,102]*12)]

swissroll_k1multiseed_parallel_batch2 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
    match_edges = ['symmetric'],
    k = [1,2,4,8],
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
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw, seed in zip(list(np.repeat([i for i in np.logspace(1,12,num=12,base = 2.0)],4)),[143,157,193,265]*12)]

swissroll_k1multiseed_parallel_batch3 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw, seed in zip(list(np.repeat([i for i in np.logspace(1,12,num=12,base = 2.0)],4)),[293,312,376,577]*12)]

swissroll_k1multiseed_parallel_batch4 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=random.sample([int(i) for i in np.logspace(3,9,num=7,base = 2.0)], 7),
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
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
        k_min = 10,
        k_max = 30,
        k_step = 5,
    )],
    uid = [''],
    toposig_kwargs=[dict()],
    method_args=[dict(n_jobs = 1)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw, seed in zip(list(np.repeat([i for i in np.logspace(1,12,num=12,base = 2.0)],4)),[600,654,789,872]*12)]

#######
euler_kn_seed1_parallel_push1 = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4, 9, base=2, num=6)],
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[tlw],
    match_edges = ['push1'],
    k = [2,4,8,16],
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
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/push1/kn_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for tlw in [int(i) for i in np.logspace(0, 11, base=2, num=12)]]

euler_kn_seed1_parallel = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4, 9, base=2, num=6)],
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 9, base=2, num=10)],
    match_edges = ['symmetric'],
    k = [n],
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
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/kn_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for n in [2,4,8,16]]


euler_k1_seed1 = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4, 9, base=2, num=6)],
    n_epochs=[500],
    weight_decay=[0],
    early_stopping=[20],
    rec_loss_weight=[1],
    top_loss_weight=[int(i) for i in np.logspace(0, 9, base=2, num=10)],
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
    method_args=[dict(n_jobs = 4)],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/SwissRoll/k1_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)



### TEST
swissroll_testing = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[512],
    n_epochs=[2],
    weight_decay=[0],
    early_stopping=[15],
    rec_loss_weight=[1],
    top_loss_weight=[512],
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
    method_args=[dict(n_jobs =1,online_wc = True)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/verification',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
)

swissroll_testing_verification = ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[15],
    rec_loss_weight=[1],
    top_loss_weight=[256],
    match_edges = ['verification'],
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
    method_args=[dict(n_jobs =1, verification = True)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/verification',
    seed = 1,
    device = 'cpu',
    num_threads=2,
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
    match_edges = ['push1'],
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
    method_args=[dict(n_jobs = 8)],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/verification',
    seed = 1,
    device = 'cpu',
    num_threads=8,
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


swissroll_testing_euler_parallel = [ConfigGrid_TopoAE_ext(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[5],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[i],
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
) for i in [1,2]]