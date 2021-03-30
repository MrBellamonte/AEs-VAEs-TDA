from src.datasets.datasets import MNIST_offline
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_MNIST_SMALL

mnist_test512_cuda = ConfigGrid_WCAE(
    learning_rate=[1/100],
    batch_size=[512],
    n_epochs=[1],
    weight_decay=[1e-6],
    early_stopping=[50],
    rec_loss_weight=[1],
    top_loss_weight=[1,2],
    match_edges=['push_active'],
    k=[1],
    r_max=[10],
    model_class=[ConvAE_MNIST_SMALL],
    model_kwargs=[dict()],
    dataset=[MNIST_offline()],
    sampling_kwargs=[dict(root_path = '/content/gdrive/My Drive/MT_projectfolder/MT/AEs-VAEs-TDA')],
    eval=[ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5,
    )],
    uid=[''],
    toposig_kwargs=[dict()],
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], wc_offline = [dict(path_to_data = '/content/gdrive/My Drive/MT_projectfolder/MT/AEs-VAEs-TDA/src/datasets/WitnessComplexes/mnist/MNIST_offline-bs512-seed838-noiseNone-ced06774')]),
    experiment_dir='/output/WAE/mnist_precomputed_2',
    seed=838,
    device='cuda',
    num_threads=1,
    verbose=True,
)