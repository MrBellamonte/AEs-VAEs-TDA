import numpy as np

from src.datasets.datasets import Spheres, SwissRoll
from src.evaluation.config import ConfigEval
from src.models.TopoAE.config import ConfigGrid_TopoAE, ConfigTopoAE
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae

### TEST AND PLACEHOLDER CONFIGURATIONS

test_grid_local = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[8,16],
    n_epochs=[1],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [64]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = True,
        k_min=5,
        k_max=105,
        k_step=25,
    )],
    uid = [''],
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/TopoAE_testing_final_3',
    seed = 1,
    device = 'cpu',
    num_threads=2,
    verbose = False
)


test_grid_euler = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[32,64],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [50]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = True,
        k_min = 5,
        k_max = 10,
        k_step = 5,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/test/1',
    seed = 1,
    device = 'cpu',
    num_threads=2,
    verbose = False
)


placeholder_config_topoae = ConfigTopoAE(
    learning_rate=1/1000,
    batch_size=16,
    n_epochs=2,
    weight_decay=0,
    early_stopping=5,
    rec_loss_weight=1,
    top_loss_weight=1,
    toposig_kwargs = dict(match_edges = 'symmetric'),
    model_class=Autoencoder_MLP_topoae,
    model_kwargs={
        'input_dim'         : 101,
        'latent_dim'        : 2,
        'size_hidden_layers': [128, 64, 32]
    },
    dataset=Spheres(),
    sampling_kwargs={
        'n_samples': 500
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 6,
        k_step = 1,
    )],
    uid = '',
)



### SPHERES

spheres_lowmemory_lowbs_euler_seed1_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(3,9,num=7,base = 2.0)],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,4,num=5,base = 2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [640]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/Spheres/lowbs_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)

spheres_lowmemory_midbs_euler_seed1_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(10,11,num=2,base = 2.0)],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,4,num=5,base = 2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [640]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 80,
        k_step = 25,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/Spheres/lowbs_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)


### SWISSROLL
swissroll_midsize_lowbs_euler_seed1_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(3,9,num=7,base = 2.0)],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,4,num=5,base = 2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [2],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 85,
        k_step = 25,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/SwissRoll/midsize_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)


swissroll_midsize_midbs_euler_seed1_1 = ConfigGrid_TopoAE(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(10,11,num=2,base = 2.0)],
    n_epochs=[100],
    weight_decay=[0],
    early_stopping=[5],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,4,num=5,base = 2.0)],
    toposig_kwargs = [dict(match_edges = 'symmetric')],
    model_class=[Autoencoder_MLP_topoae],
    model_kwargs={
        'input_dim'         : [2],
        'latent_dim'        : [2],
        'size_hidden_layers': [[32, 32]]
    },
    dataset=[SwissRoll()],
    sampling_kwargs={
        'n_samples': [2530]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = 'test',
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min = 5,
        k_max = 85,
        k_step = 25,
    )],
    uid = [''],
    experiment_dir='/cluster/home/schsimo/MT/output/TopoAE/SwissRoll/midsize_seed1',
    seed = 1,
    device = 'cpu',
    num_threads=1,
    verbose = False
)


# test_grid2 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[16,32,64],
#     n_epochs=[100],
#     weight_decay=[0],
#     early_stopping=[5],
#     rec_loss_weight=[1],
#     top_loss_weight=[4,1,1/4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[32, 16]]
#     },
#     dataset=[SwissRoll()],
#      sampling_kwargs={
#          'n_samples': [4608]
#      },
#     eval=[ConfigEval(
#         active = True,
#         evaluate_on = 'test',
#         save_eval_latent = True,
#         save_train_latent = True,
#         k_min = 5,
#         k_max = 10,
#         k_step = 5,
#     )],
#     uid = ['']
# )


# Spheres
#
# moor_config_approx_1 = ConfigGrid_TopoAE(
#     learning_rate=[27/100000],
#     batch_size=[28],
#     n_epochs=[100],
#     weight_decay=[1e-05],
#     rec_loss_weight=[1],
#     top_loss_weight=[float(Fraction(22/51))],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[32, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [500]
#     }
# )
#
#
# # Swiss Roll
# swiss_roll_nonoise_benchmark_1 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size= [int(i) for i in np.logspace(2,10,num=9,base = 2.0)] + [1536],
#     n_epochs=[40],
#     weight_decay=[None],
#     rec_loss_weight=[1],
#     top_loss_weight=[0],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
#
#
# swiss_roll_nonoise_1 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[2, 4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
# swiss_roll_nonoise_2 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/2, 1],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
# swiss_roll_nonoise_3 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/8, 1/4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
# swiss_roll_nonoise_4 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[1536] + [int(i) for i in np.logspace(2,10,num=9,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/32, 1/16],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
#
# swiss_roll_nonoise_5 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size= [int(i) for i in np.logspace(2,9,num=9,base = 2.0)] + [1536],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[8,16],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [3],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[16, 8]]
#     },
#     dataset=[SwissRoll()],
#     sampling_kwargs={
#         'n_samples': [1536]
#     }
# )
#
# ### All Models run on Euler
#
#
# ## FIRST RUNS
# # regular batch sizes
# eulergrid_280720 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/4,1/2,1,2,4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [500]
#     }
# )
#
# # medium batch sizes
# eulergrid_280720_2 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[int(i) for i in np.logspace(10,12,num=3,base = 2.0)],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/4,1/2,1,2,4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [500]
#     }
# )
#
# # large batch sizes
# eulergrid_280720_3 = ConfigGrid_TopoAE(
#     learning_rate=[1/1000],
#     batch_size=[8192,10000],
#     n_epochs=[40],
#     weight_decay = [None],
#     rec_loss_weight=[1],
#     top_loss_weight=[1/4,1/2,1,2,4],
#     toposig_kwargs = [dict(match_edges = 'symmetric')],
#     model_class=[Autoencoder_MLP_topoae],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [500]
#     }
# )