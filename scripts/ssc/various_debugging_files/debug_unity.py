import torch
from torch import nn
from torchsummary import summary

from src.datasets.datasets import Unity_Rotblock
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext
from src.models.autoencoder.autoencoders import ConvAElarge_Unity

if __name__ == "__main__":

    unity_test = ConfigGrid_WCAE(
        learning_rate=[1/100],
        batch_size=[120],
        n_epochs=[200],
        weight_decay=[1e-6],
        early_stopping=[20],
        rec_loss_weight=[1],
        top_loss_weight=[512,1024],
        match_edges=['push_active'],
        k=[1],
        r_max=[10],
        model_class=[ConvAElarge_Unity],
        model_kwargs=[dict()],
        dataset=[Unity_Rotblock()],
        sampling_kwargs=[dict()],
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
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                         wc_offline=[dict(path_to_data='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/block_rotation_1')]),
        experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/testing_unity',
        seed=1,
        device='cpu',
        num_threads=4,
        verbose=True,
    )



    simulator_TopoAE_ext(unity_test)



