import datetime
import os

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.shapes import dsphere
from src.model.train_engine import train, train_teststructure

if __name__ == "__main__":


    config = {
        'train_args': {
            'learning_rate': 0.001,
            'batch_size'   : 512,
            'n_epochs'     : 5,
            'rec_loss_w'   : 1.0,
            'top_loss_w'   : 1.0,
        },
        'model_args': {
            'class_id': 'autoencoder',
            'kwargs'  : {
                'input_dim': 4,
                'latent_dim' : 2,
                'size_hidden_layers': [128,64,32]
                }
            }
        }


    X, y = dsphere(n=1024*4, r=10, d=3)


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/todelete'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass



    train_teststructure(dataset, config, path)
