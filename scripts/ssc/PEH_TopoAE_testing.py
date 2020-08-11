import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.datasets.datasets import SwissRoll
from src.models.TopoAE.topology import PersistentHomologyCalculation


def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None]-x_flat, dim=2, p=p)
    return distances

PATH_ROOT = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/'
def make_data(data, color, name = '', path_root = PATH_ROOT):
    data_torch = torch.Tensor(data)
    pers_calc = PersistentHomologyCalculation()

    data_distances = _compute_distance_matrix(data_torch)
    pairings = pers_calc(data_distances)

    print(type(pairings[0]))
    print(type(data))

    path_pairings = '{}pairings_{}.npy'.format(path_root, name)
    path_data = '{}data_{}.npy'.format(path_root, name)
    path_color = '{}color_{}.npy'.format(path_root, name)
    np.save(path_pairings,pairings[0])
    np.save(path_data, data)
    np.save(path_color, color)



def make_plot(data, pairings, color):
    ax = plt.gca(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=100)

    for pairing in pairings:
        ax.plot([data[pairing[0], 0], data[pairing[1], 0]],
                [data[pairing[0], 1], data[pairing[1], 1]],
                [data[pairing[0], 2], data[pairing[1], 2]], color='r')



    ax.view_init(15, 90)
    plt.show()




if __name__ == "__main__":
    dataset_sampler = SwissRoll()
    data, color = dataset_sampler.sample(1000)
    make_data(data, color, name = 'test')
    name = 'test'

    path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)

    #
    make_plot(data, pairings, color)