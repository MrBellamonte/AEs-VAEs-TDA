from src.competitors.competitor_models import tSNE
from src.competitors.config import ConfigGrid_Competitors
from src.datasets.datasets import SwissRoll
from src.evaluation.config import ConfigEval

swissroll_test = ConfigGrid_Competitors(
    model_class = [tSNE],
    model_kwargs=[dict(n_jobs = 4)],
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
    experiment_dir='/Users/simons/PycharmProjects/MT-VAEs-TDA/output/competitors/testing',
    seed = 1,
    verbose = True
)


swissroll_euler = [ConfigGrid_Competitors(
    model_class = [tSNE],
    model_kwargs=[dict(n_jobs = 1, perplexity = p) for p in [10,20,30,40,50,60,70,80,90,100]],
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
    experiment_dir='/cluster/scratch/schsimo/output/tsne_final',
    seed = seed,
    verbose = True
) for seed in [480, 367, 887, 718, 672, 172,  12, 326, 910, 688]]
