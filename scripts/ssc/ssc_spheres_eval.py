import pickle

from torch import Tensor
from torch.utils.data import TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.datasets import create_sphere_dataset
from src.datasets.shapes import dsphere
from src.model.eval_engine import get_model, get_latentspace_representation

if __name__ == "__main__":


    X, y = create_sphere_dataset()

    dataset = TensorDataset(Tensor(X), Tensor(y))

    path0 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw0-4a9755a1/'
    path1 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw3-c1c53ae9/'
    path2 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw6-a211b361/'
    path3 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw12-939f0a1e/'
    path4 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw18-1f7179f4/'
    path5 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw25-8f0548ab/'
    path6 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw50-0b31e4fd/'
    path7 = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/spheres_default/2020-06-30/autoencoder-128-64-32-2-lr1-bs256-nep50-rlw100-tlw100-da7b5292/'

    paths = [path0,path1, path2, path3,path4,path5, path6, path7]

    for path in paths:
        model = get_model(path)

        X,Y,Z = get_latentspace_representation(model,dataset)

        plt.scatter(Z[:, 0], Z[:, 1], c=Y,
                    cmap=plt.cm.Spectral, s=2., alpha=0.5)

        plt.show()
