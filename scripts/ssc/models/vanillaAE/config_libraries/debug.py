from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae
from src.models.vanillaAE.config import ConfigGrid_VanillaAE

ae_test = ConfigGrid_VanillaAE(
    learning_rate=[1/10, 1/100, 1/1000],
    batch_size=[64],
    n_epochs=[1000],
    early_stopping=[50],
    weight_decay=[1e-6],
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
    method_args=[dict()],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/vanillaAE/test',
    seed=1,
    device='cpu',
    num_threads=1,
    verbose=False,
)
