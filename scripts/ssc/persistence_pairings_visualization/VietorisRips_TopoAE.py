import numpy as np
import torch

from scripts.ssc.persistence_pairings_visualization.utils_definitions import (
    PATH_ROOT_SWISSROLL,
    make_plot)
from src.datasets.datasets import SwissRoll
from src.models.TopoAE.topology import PersistentHomologyCalculation


def _compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None]-x_flat, dim=2, p=p)
    return distances

def make_data(data, color, name = '', path_root = PATH_ROOT_SWISSROLL):
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




if __name__ == "__main__":
    dataset_sampler = SwissRoll()
    n_samples_array = [32,48,64,96,128]
    seeds = [10,13,20]

    for seed in seeds:
        for n_samples in n_samples_array:

            name = 'vr_ns{}_seed{}'.format(n_samples, seed)

            data, color = dataset_sampler.sample(n_samples, seed = seed)
            make_data(data, color, name = name)

            path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            path_data = '{}data_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            path_color = '{}color_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
            #
            make_plot(data, pairings, color, name = name)

    # name = '512_1'
    # data, color = dataset_sampler.sample(512)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)
    #
    # name = '256_1'
    # data, color = dataset_sampler.sample(256)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)
    #
    # name = '128_1'
    # data, color = dataset_sampler.sample(128)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)
    #
    # name = '64_1'
    # data, color = dataset_sampler.sample(64)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)
    #
    # name = '32_1'
    # data, color = dataset_sampler.sample(32)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)
    #
    # name = '16_1'
    # data, color = dataset_sampler.sample(16)
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    #
    # #
    # make_plot(data, pairings, color, name = name)