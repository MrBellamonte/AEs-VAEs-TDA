from src.datasets.datasets import Unity_RotOpenAI
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_Unity480320


# 36
rotopenai_cluster = ConfigGrid_WCAE(
        learning_rate=[1/10,1/1000,1/100],
        batch_size=[180],
        n_epochs=[1000],
        weight_decay=[1e-6],
        early_stopping=[50],
        rec_loss_weight=[1],
        top_loss_weight=[1],
        match_edges=['push_active'],
        k=[1,2,3,4,5],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotOpenAI()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=False,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125,1.25], online_wc=[True], dist_x_land = [True],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
        experiment_dir='/cluster/scratch/schsimo/output/openai_rot',
        seed=1,
        device='cpu',
        num_threads=1,
        verbose=True,
    )