from src.competitors.competitor_models import tSNE
from src.competitors.config import ConfigGrid_Competitors
from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae



WCAE_sample_config = ConfigGrid_WCAE(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[5],
    weight_decay=[0],
    early_stopping=[35],
    rec_loss_weight=[1],
    top_loss_weight=[1024],
    match_edges = ['push_active'],
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
    method_args=[dict(n_jobs = 1, normalize = True)],
    experiment_dir='output/sample',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = True,
)


TopoAE_sample_config = ConfigGrid_TopoAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[1024],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
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
    uid = [''],
    method_args = [None],
    experiment_dir='output/sample',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)


tSNE_sample_config = ConfigGrid_Competitors(
    model_class = [tSNE],
    model_kwargs=[dict(n_jobs = 1)],
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = None,
        eval_manifold=True,
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid = [''],
    experiment_dir='output/sample',
    seed = 1,
    verbose = True
)