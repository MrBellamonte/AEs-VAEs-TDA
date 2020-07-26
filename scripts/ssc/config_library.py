from src.datasets.datasets import Spheres
from src.models.COREL.config import ConfigGrid_COREL
from src.models.autoencoders import autoencoder
from src.models.loss_collection import L1Loss, TwoSidedHingeLoss, HingeLoss



test_run_leonhard = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[64, 128],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[L1Loss()],
    top_loss_weight=[1],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [250]
    }
)

config_grid_Spheres_n3_250_l1 = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[8,16,32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[L1Loss()],
    top_loss_weight=[1/64,1/32,1/16,1/8,1/4,1/2,1,2,4],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres(n_spheres=3)],
    sampling_kwargs={
        'n_samples': [250]
    }
)


config_grid_Spheres_n3_250_tshinge = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[8,16,32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[TwoSidedHingeLoss(ratio=1/2),TwoSidedHingeLoss(ratio=1/4)],
    top_loss_weight=[1/64,1/32,1/16,1/8,1/4,1/2,1,2,4],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres(n_spheres=3)],
    sampling_kwargs={
        'n_samples': [250]
    }
)



config_grid_Spheres_L1 = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[L1Loss()],
    top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)


config_grid_Spheres_benchmark = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[L1Loss()],
    top_loss_weight=[0],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)

config_grid_Spheres_Hinge = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[HingeLoss(), HingeLoss(penalty_type='squared')],
    top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1,2,4,8,16,32],
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)


config_grid_Spheres_TwoSidedHinge = ConfigGrid_COREL(
    learning_rate=[1/1000],
    batch_size=[32, 64, 128, 256, 512],
    n_epochs=[40],
    rec_loss=[L1Loss()],
    rec_loss_weight=[1],
    top_loss=[TwoSidedHingeLoss(), TwoSidedHingeLoss(penalty_type='squared'),TwoSidedHingeLoss(ratio=1/4), TwoSidedHingeLoss(ratio=1/4,penalty_type='squared')],
    top_loss_weight=[1/2048, 1/1024,1/512,1/256,1/128,1/64,1/32,1/16,1/8], #[1/4,1/2,1,2,4,8,16,32]
    model_class=[autoencoder],
    model_kwargs={
        'input_dim'         : [101],
        'latent_dim'        : [2],
        'size_hidden_layers': [[128, 64, 32]]
    },
    dataset=[Spheres()],
    sampling_kwargs={
        'n_samples': [500]
    }
)




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
        'model_class': [autoencoder],
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
        'model_class': [autoencoder],
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