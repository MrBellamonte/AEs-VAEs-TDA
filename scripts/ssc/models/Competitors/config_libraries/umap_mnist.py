from src.competitors.competitor_models import UMAP
from src.competitors.config import ConfigGrid_Competitors
from src.datasets.datasets import SwissRoll, MNIST_offline
from src.evaluation.config import ConfigEval

umap_mnist_euler_1 = ConfigGrid_Competitors(
    model_class = [UMAP],
    model_kwargs=dict(n_neighbors = [15,20,25,30,35,40,45,50,55,65,70], min_dist = [0.05,0.1,0.15,0.2,0.25]),
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=False,
        online_visualization=False,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid = [''],
    experiment_dir='/cluster/scratch/schsimo/output/umap_mnist',
    seed = 1,
    verbose = True
)