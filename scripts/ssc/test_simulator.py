import datetime
import os

from torch import Tensor
from torch.utils.data import TensorDataset

from src.datasets.datasets import Spheres
from src.models.COREL.train_engine import simulator

from scripts.ssc.config_library import *


if __name__ == "__main__":

    X, y = Spheres()


    dataset = TensorDataset(Tensor(X), Tensor(y))
    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/bugfix/1'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    for config_grid in [config_grid_Spheres_n3_250_l1]:
        simulator(config_grid, path, verbose = True)
