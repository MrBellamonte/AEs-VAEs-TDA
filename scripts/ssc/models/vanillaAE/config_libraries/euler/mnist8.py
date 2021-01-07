import itertools

from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import DeepAE_MNIST_8D
from src.models.vanillaAE.config import ConfigGrid_VanillaAE

mnist_1_deepae8 = [ConfigGrid_VanillaAE(
    learning_rate=[1/10,1/100,1/1000],
    batch_size=[64,128,256,512],
    n_epochs=[1000],
    weight_decay=[1e-6],
    early_stopping=[32],
    model_class=[DeepAE_MNIST_8D],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path='/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=False,
        save_train_latent=False,
        online_visualization=False,
        quant_eval=True,
        k_min=4,
        k_max=16,
        k_step=4)],
    uid=[''],
    method_args=[dict()],
    experiment_dir='/cluster/scratch/schsimo/output/mnist_ae_1_deepae8',
    seed=seed,
    device='cpu',
    num_threads=1,
    verbose=False) for seed in [838,579,1988]]

mnist_1_deepae8_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in mnist_1_deepae8]))