from src.datasets.datasets import Unity_XYTransOpenAI
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE
from src.models.autoencoder.autoencoders import ConvAE_Unity480320

unity_xytrans_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000,1/100,1/10],
    batch_size=[200,400],
    n_epochs=[5000],
    weight_decay=[0],
    early_stopping=[250],
    rec_loss_weight=[1],
    top_loss_weight=[1,2,4],
    toposig_kwargs=[dict(match_edges='symmetric')],
    model_class=[ConvAE_Unity480320],
    model_kwargs=[dict()],
    dataset=[Unity_XYTransOpenAI(version='xy_trans_l_newpers')],
    sampling_kwargs=[dict(root_path='/cluster/scratch/schsimo')],
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
    method_args=[dict(val_size = 0)],
    experiment_dir='/cluster/scratch/schsimo/output/TopoAE_xytrans1',
    seed=1,
    device='cpu',
    num_threads=1,
    verbose=False
)
