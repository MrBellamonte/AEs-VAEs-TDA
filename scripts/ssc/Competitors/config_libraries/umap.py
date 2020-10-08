from src.competitors.competitor_models import UMAP
from src.competitors.config import ConfigGrid_Competitors
from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval

swissroll_test = ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=[dict()],
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
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/competitors/testing',
    seed = 1,
    verbose = True
)

swissroll_umap_grid = [ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=[dict(n_neighbors = nn, min_dist = 0.3+mdist) for nn in [2,4,8,10,12,16,20,24,28,32]],
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = None,
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid = [''],
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/competitors/swissroll_umap',
    seed = 100,
    verbose = True
) for mdist in [0.05,0.1,0.15,0.2, 0.25, 0.3]]


swissroll_euler = [ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=dict(n_neighbors = [15,20,25,30,35,40,45,50,55,65,70], min_dist = [0.05,0.1,0.15,0.2,0.25]),
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
        k_min=15,
        k_max=45,
        k_step=15,
    )],
    uid = [''],
    experiment_dir='/cluster/scratch/schsimo/output/umap_final',
    seed = seed,
    verbose = True
) for seed in [480, 367, 887, 718, 672, 172,  12, 326, 910, 688]]