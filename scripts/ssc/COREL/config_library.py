from fractions import Fraction

import numpy as np

from src.datasets.datasets import Spheres
from src.evaluation.config import ConfigEval
from src.models.COREL.config import ConfigGrid_COREL, ConfigCOREL
from src.models.autoencoder.autoencoders import Autoencoder_MLP
from src.models.loss_collection import L1Loss, TwoSidedHingeLoss, HingeLoss


placeholder_config_corel = ConfigCOREL(
    learning_rate=1/1000,
    batch_size=16,
    n_epochs=2,
    weight_decay=0,
    early_stopping=5,
    rec_loss=L1Loss(),
    top_loss=L1Loss(),
    rec_loss_weight=1,
    top_loss_weight=1,
    model_class=Autoencoder_MLP,
    model_kwargs={
        'input_dim'         : 101,
        'latent_dim'        : 2,
        'size_hidden_layers': [128, 64, 32]
    },
    dataset=Spheres(),
    sampling_kwargs={
        'n_samples': 500
    },
    eval=ConfigEval(
        active=True,
        evaluate_on='test',
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=True,
        k_min=5,
        k_max=105,
        k_step=25,
    ),
    uid = '',
)

test_grid_local = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[64],
    n_epochs=[20],
    weight_decay=[10e-5],
    early_stopping=[5],
    rec_loss=[L1Loss()],
    top_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss_weight=[1],
    model_class=[Autoencoder_MLP],
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
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/TopoAE_testing_COREL',
    seed = 1,
    verbose = False
)



grid_spheres = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(6,9,num=4,base = 2.0)],#   [int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
    n_epochs=[100],
    weight_decay=[10e-5],
    early_stopping=[5],
    rec_loss=[L1Loss()],
    top_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-8,0,num=9,base = 2.0)],
    model_class=[Autoencoder_MLP],
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
        online_visualization = True,
        k_min=5,
        k_max=105,
        k_step=25,
    )],
    uid = [''],
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/output/TopoAE/Spheres/l1',
    seed = 1,
    verbose = False
)

grid_spheres_ts = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
    n_epochs=[100],
    weight_decay=[10e-5],
    early_stopping=[5],
    rec_loss=[L1Loss()],
    top_loss=[TwoSidedHingeLoss(ratio=1/4)],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-4,0,num=5,base = 2.0)], #[i for i in np.logspace(-8,0,num=9,base = 2.0)],
    model_class=[Autoencoder_MLP],
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
        k_min=5,
        k_max=105,
        k_step=25,
    )],
    uid = [''],
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/output/TopoAE/Spheres/ts',
    seed = 1,
    verbose = False
)


grid_spheres_ts_sq = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[32],
    n_epochs=[100],
    weight_decay=[10e-5],
    early_stopping=[10],
    rec_loss=[L1Loss()],
    top_loss=[TwoSidedHingeLoss(ratio=1/4, penalty_type= 'squared')],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-6,-4,num=3,base = 2.0)], #[i for i in np.logspace(-8,0,num=9,base = 2.0)],
    model_class=[Autoencoder_MLP],
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
        k_min=5,
        k_max=105,
        k_step=25,
    )],
    uid = [''],
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/output/TopoAE/Spheres/ts_sq',
    seed = 1,
    verbose = False
)


grid_spheres_ts_large = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[int(i) for i in np.logspace(4,9,num=6,base = 2.0)],
    n_epochs=[100],
    weight_decay=[10e-5],
    early_stopping=[10],
    rec_loss=[L1Loss()],
    top_loss=[TwoSidedHingeLoss(ratio=1/4)],
    rec_loss_weight=[1],
    top_loss_weight=[i for i in np.logspace(-8,0,num=9,base = 2.0)],
    model_class=[Autoencoder_MLP],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
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
        k_min=5,
        k_max=105,
        k_step=25,
    )],
    uid = [''],
    experiment_dir='/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/output/TopoAE/Spheres/ts_large2',
    seed = 2,
    verbose = False
)
# conifg_spheres_fullbatch2_l1 = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[25,50,100,250,500],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[TwoSidedHingeLoss(ratio=1/4)],
#     top_loss_weight=[float(Fraction(1/i))for i in np.logspace(-2,9,num=12,base = 2.0)],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [25]
#     }
# )

# conifg_spheres_fullbatch2_tshinge = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[25,50,100,250,500],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[TwoSidedHingeLoss(ratio=1/4)],
#     top_loss_weight=[float(Fraction(1/i))for i in np.logspace(-2,9,num=12,base = 2.0)],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [25]
#     }
# )
#
#
#
# conifg_spheres_fullbatch_l1 = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     #batch_size=[int(i) for i in np.logspace(3,9,num=7,base = 2.0)],
#     batch_size=[500],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[L1Loss()],
#     top_loss_weight=[float(Fraction(1/i))for i in np.logspace(-2,9,num=12,base = 2.0)],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [25]
#     }
# )
#
#
#
# test_run_leonhard = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[64, 128],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[L1Loss()],
#     top_loss_weight=[1],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres()],
#     sampling_kwargs={
#         'n_samples': [250]
#     }
# )
#
# config_grid_Spheres_n3_250_l1 = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[8,16,32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[L1Loss()],
#     top_loss_weight=[1/64,1/32,1/16,1/8,1/4,1/2,1,2,4],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres(n_spheres=3)],
#     sampling_kwargs={
#         'n_samples': [250]
#     }
# )
#
#
# config_grid_Spheres_n3_250_tshinge = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[8,16,32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[TwoSidedHingeLoss(ratio=1/2),TwoSidedHingeLoss(ratio=1/4)],
#     top_loss_weight=[1/64,1/32,1/16,1/8,1/4,1/2,1,2,4],
#     model_class=[Autoencoder_MLP],
#     model_kwargs={
#         'input_dim'         : [101],
#         'latent_dim'        : [2],
#         'size_hidden_layers': [[128, 64, 32]]
#     },
#     dataset=[Spheres(n_spheres=3)],
#     sampling_kwargs={
#         'n_samples': [250]
#     }
# )
#
#
#
# config_grid_Spheres_L1 = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[L1Loss()],
#     top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32],
#     model_class=[Autoencoder_MLP],
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
#
# config_grid_Spheres_benchmark = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[L1Loss()],
#     top_loss_weight=[0],
#     model_class=[Autoencoder_MLP],
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
# config_grid_Spheres_Hinge = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[HingeLoss(), HingeLoss(penalty_type='squared')],
#     top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32],
#     model_class=[Autoencoder_MLP],
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
#
# config_grid_Spheres_TwoSidedHinge = ConfigGrid_COREL(
#     learning_rate=[1/1000],
#     batch_size=[32, 64, 128, 256, 512],
#     n_epochs=[40],
#     rec_loss=[L1Loss()],
#     rec_loss_weight=[1],
#     top_loss=[TwoSidedHingeLoss(), TwoSidedHingeLoss(penalty_type='squared'),TwoSidedHingeLoss(ratio=1/4), TwoSidedHingeLoss(ratio=1/4,penalty_type='squared')],
#     top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8], #[1/4,1/2,1,2,4,8,16,32]
#     model_class=[Autoencoder_MLP],
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




# OLD CONFIGS


config_grid_testSpheres = {
    'train_args': {
        'learning_rate': [1/1000],
        'batch_size'   : [32,64,128],
        'n_epochs'     : [2],
        'rec_loss' : {
            'loss_class' : [L1Loss()],
            'weight' : [1]
        },
        'top_loss': {
            'loss_class': [L1Loss()],
            'weight'    : [1]
        },
    },
    'model_args': {
        'model_class': [Autoencoder_MLP],
        'kwargs'  : {
            'input_dim'         : [101],
            'latent_dim'        : [2],
            'size_hidden_layers': [[128 ,64 ,32]]
        }
    },
    'data_args' :{
        'dataset' : Spheres(),
        'kwargs' :{
            'n_samples': 500
        }
    }
}


config_grid_test_tshinge = {
    'train_args': {
        'learning_rate': [1/1000],
        'batch_size'   : [32,64,128],
        'n_epochs'     : [2],
        'rec_loss' : {
            'loss_class' : [L1Loss()],
            'weight' : [1]
        },
        'top_loss': {
            'loss_class': [TwoSidedHingeLoss()],
            'weight'    : [1]
        },
    },
    'model_args': {
        'model_class': [Autoencoder_MLP],
        'kwargs'  : {
            'input_dim'         : [101],
            'latent_dim'        : [2],
            'size_hidden_layers': [[128 ,64 ,32]]
        }
    },
    'data_args' :{
        'dataset' : Spheres(),
        'kwargs' :{
            'n_samples': 500
        }
    }
}