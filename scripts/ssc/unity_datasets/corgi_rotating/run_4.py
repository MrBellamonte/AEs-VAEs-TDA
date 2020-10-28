import torch
from torch import nn
from torchsummary import summary

from src.datasets.datasets import Unity_Rotblock, Unity_RotCorgi
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext
from src.models.autoencoder.autoencoders import ConvAElarge_Unity, ConvAE_Unity480320

if __name__ == "__main__":

    corgi_1 = ConfigGrid_WCAE(
        learning_rate=[1/100],
        batch_size=[30],
        n_epochs=[200],
        weight_decay=[1e-6],
        early_stopping=[5],
        rec_loss_weight=[1],
        top_loss_weight=[1024],
        match_edges=['push_active'],
        k=[1],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(version = 4, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True], lam_t_bi = [[0,1]],
                         wc_offline=[dict(path_to_data='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/corgi_rotation_4_l')]),
        experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/corgi/rotating_4_l',
        seed=1,
        device='cpu',
        num_threads=2,
        verbose=True,
    )

    simulator_TopoAE_ext(corgi_1)





