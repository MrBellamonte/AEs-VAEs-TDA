import itertools

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

rotopenai_cluster_decay = ConfigGrid_WCAE(
    learning_rate=[1/10, 1/1000, 1/100],
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
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=10,
        k_step=5,
        quant_eval=False

    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125, 1.25], online_wc=[True],
                     dist_x_land=[True],
                     lam_t_decay=[
                         {0  : 1024, 25: 512, 50: 256, 75: 128, 100: 64, 125: 32, 150: 16, 175: 8,
                          200: 4, 250: 2, 300: 1},
                         {0  : 512, 25: 256, 50: 128, 75: 64, 100: 32, 125: 16, 150: 8, 175: 4,
                          200: 2, 250: 1},
                         {0: 256, 25: 128, 50: 64, 75: 32, 125: 16, 150: 8, 175: 4, 200: 2, 250: 1},
                         {0: 128, 25: 64, 50: 32, 75: 16, 125: 8, 150: 4, 175: 2, 200: 1}],
                     wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
    experiment_dir='/cluster/scratch/schsimo/output/rotating_decay',
    seed=1,
    device='cpu',
    num_threads=1,
    verbose=False,
)





#### LEONHARD
leo_rotopenai_decay1 = ConfigGrid_WCAE(
    learning_rate=[1/1000, 1/100],
    batch_size=[180],
    n_epochs=[12000],
    weight_decay=[0],
    early_stopping=[150],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    match_edges=['push_active'],
    k=[2,3],
    r_max=[10],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_RotOpenAI()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
        quant_eval=False

    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     dist_x_land=[True],
                     lam_t_decay=[
                         {0  : 1024, 25: 512, 50: 256, 75: 128, 100: 64, 125: 32, 150: 16, 175: 8,
                          200: 4, 250: 2, 300: 1},
                         {0  : 512, 25: 256, 50: 128, 75: 64, 100: 32, 125: 16, 150: 8, 175: 4,
                          200: 2, 250: 1},
                         {0: 256, 25: 128, 50: 64, 75: 32, 125: 16, 150: 8, 175: 4, 200: 2, 250: 1}],
                     wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
    experiment_dir='/cluster/scratch/schsimo/output/rotating_decay',
    seed=1,
    device='cuda',
    num_threads=1,
    verbose=False,
)



leo_rotopenai_notopo = ConfigGrid_WCAE(
    learning_rate=[1/1000, 1/100],
    batch_size=[180],
    n_epochs=[12000],
    weight_decay=[0],
    early_stopping=[150],
    rec_loss_weight=[1],
    top_loss_weight=[0],
    match_edges=['push_active'],
    k=[1],
    r_max=[10],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_RotOpenAI()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
        quant_eval=False

    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True],
                     dist_x_land=[True],
                     wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
    experiment_dir='/cluster/scratch/schsimo/output/rotating_decay',
    seed=1,
    device='cuda',
    num_threads=1,
    verbose=False,
)

leo_rotopenai_decay2 = ConfigGrid_WCAE(
    learning_rate=[1/1000, 1/100],
    batch_size=[180],
    n_epochs=[12000],
    weight_decay=[0],
    early_stopping=[125],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_RotOpenAI()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
        quant_eval=False

    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     dist_x_land=[True],
                     lam_t_decay=[
                         {0  : 1024, 25: 512, 50: 256, 75: 128, 100: 64, 125: 32, 150: 16, 175: 8,
                          200: 4, 250: 2, 300: 1, 350: 0.5, 400: 0.25, 450: 1/8, 500: 1/16, 550: 2**5, 600: 2**6, 650: 2**7,700: 2**8, 800: 2**10, 900: 2**12, 1000: 2**14, 1100: 2**16, 1200: 2**18},],
                     wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
    experiment_dir='/cluster/scratch/schsimo/output/rotating_decay_2',
    seed=2,
    device='cuda',
    num_threads=1,
    verbose=False,
)

leo_rotopenai_decay3 = ConfigGrid_WCAE(
    learning_rate=[1/1000, 1/100],
    batch_size=[180],
    n_epochs=[12000],
    weight_decay=[0],
    early_stopping=[125],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    match_edges=['push_active'],
    k=[3],
    r_max=[10],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_RotOpenAI()],
    sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
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
        quant_eval=False

    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1, 1.125], online_wc=[True],
                     dist_x_land=[True],
                     lam_t_decay=[
                         {0  : 1024, 25: 512, 50: 256, 75: 128, 100: 64, 125: 32, 150: 16, 175: 8,
                          200: 4, 250: 2, 300: 1, 350: 0.5, 400: 0.25, 450: 1/8, 500: 1/16, 550: 2**5, 600: 2**6, 650: 2**7,700: 2**8, 800: 2**10, 900: 2**12, 1000: 2**14, 1100: 2**16, 1200: 2**18},],
                     wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/openai_rotating')]),
    experiment_dir='/cluster/scratch/schsimo/output/rotating_decay_2',
    seed=2,
    device='cuda',
    num_threads=1,
    verbose=False,
)

leo_rotopenai_1 = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in [leo_rotopenai_decay1,leo_rotopenai_notopo]]))
