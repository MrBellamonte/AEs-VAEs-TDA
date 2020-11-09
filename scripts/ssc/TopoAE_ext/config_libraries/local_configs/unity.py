from src.datasets.datasets import Unity_RotOpenAI
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_Unity480320

rotopenai_test = ConfigGrid_WCAE(
        learning_rate=[1/100],
        batch_size=[180],
        n_epochs=[1000],
        weight_decay=[1e-6],
        early_stopping=[30],
        rec_loss_weight=[1],
        top_loss_weight=[1/8,1/4,1/2,1,0],
        match_edges=['push_active'],
        k=[1],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotOpenAI()],
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
                         wc_offline=[dict(path_to_data='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
        experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/rotating',
        seed=1,
        device='cpu',
        num_threads=1,
        verbose=True,
    )


rotopenai_trial = ConfigGrid_WCAE(
        learning_rate=[1/10,1/1000,1/100],
        batch_size=[180],
        n_epochs=[1000],
        weight_decay=[1e-6],
        early_stopping=[30],
        rec_loss_weight=[1],
        top_loss_weight=[1/8,1/4,1/2,1,0],
        match_edges=['push_active'],
        k=[1],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotOpenAI()],
        sampling_kwargs=[dict()],
        eval=[ConfigEval(
            active=False,
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
                         wc_offline=[dict(path_to_data='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
        experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/rotating',
        seed=1,
        device='cpu',
        num_threads=1,
        verbose=True,
    )



