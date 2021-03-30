import numpy as np

from src.datasets.datasets import Spheres
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae

euler_run1 = [ConfigGrid_WCAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(3,11,num=10,base = 2.0)],
    n_epochs=[1000],
    weight_decay=[0],
    early_stopping=[10],
    rec_loss_weight=[1],
    top_loss_weight=[j],
    match_edges = ['symmetric'],
    k = [1],
    r_max = [10],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [512]
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
    method_args=[dict()],
    experiment_dir='/cluster/home/schsimo/MT/output/WCTopoAE/Spheres/testing',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False,
) for j in [int(i) for i in np.logspace(0, 10, base=2, num=4)]]
