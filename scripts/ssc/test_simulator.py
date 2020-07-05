import datetime
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.datasets import create_sphere_dataset, Spheres
from src.datasets.shapes import dsphere
from src.model.autoencoders import autoencoder
from src.model.loss_collection import L1Loss
from src.model.train_engine import simulator
from src.utils.config_utils import configs_from_grid

from scripts.ssc.config_library import *


if __name__ == "__main__":

    X, y = create_sphere_dataset()


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass

    simulator(config_grid_testSpheres, path, verbose = True)
