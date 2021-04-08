import itertools

from src.datasets.datasets import Unity_RotOpenAI, Unity_XYTransOpenAI
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_Unity480320
from src.models.vanillaAE.config import ConfigGrid_VanillaAE
from src.models.variational_autoencoder.varautoencoders import ConvVAE_Unity480320

xy_trans_final_1_k2 = ConfigGrid_WCAE(
    learning_rate=[1/100],
    batch_size=[200],
    n_epochs=[2000],
    weight_decay=[0],
    early_stopping=[250],
    rec_loss_weight=[1],
    top_loss_weight=[4,8,16],
    match_edges=['push_active'],
    k=[2],
    r_max=[10],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path = '/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA')],
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
    method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1,1.125], online_wc=[True],
                     dist_x_land=[True],val_size = [0],
                     wc_offline=[dict(path_to_data='/cluster/scratch/schsimo/WitnessComplex/unity/Unity_XYTransOpenAI-bs200-seed1-noiseNone-9426d882')]),
    experiment_dir='/cluster/scratch/schsimo/output/xy_trans_final',
    seed=2,
    device='cuda',
    num_threads=2,
    verbose=False,
)


xy_trans_test= ConfigGrid_VanillaAE(
    learning_rate=[1/10],
    batch_size=[441],
    n_epochs=[10000],
    weight_decay=[1e-6],
    early_stopping=[1000],
    model_class=[ConvVAE_Unity480320],
    model_kwargs={'lambda_kld': [1]},
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA')],
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
    method_args=dict(val_size=[0]),
    experiment_dir='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA/output/vanillaVAE/xy_trans_test',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=False)



xy_trans_test2 = ConfigGrid_VanillaAE(
    learning_rate=[1/10],
    batch_size=[128],
    n_epochs=[10000],
    weight_decay=[1e-6],
    early_stopping=[1000],
    model_class=[ConvVAE_Unity480320],
    model_kwargs={'lambda_kld': [1]},
    dataset=[Unity_XYTransOpenAI(version = 'xy_trans_final')],
    sampling_kwargs=[dict(root_path='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA')],
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
    method_args=dict(val_size=[0]),
    experiment_dir='/home/simonberg/PycharmProjects/MT_contd/AEs-VAEs-TDA/output/vanillaVAE/xy_trans_test',
    seed=838,
    device='cuda',
    num_threads=2,
    verbose=True)