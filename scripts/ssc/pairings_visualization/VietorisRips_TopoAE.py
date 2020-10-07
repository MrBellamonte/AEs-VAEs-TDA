import numpy as np
import torch

import matplotlib.pyplot as plt

from scripts.ssc.pairings_visualization.utils_definitions import (
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
    if path_root is not None:
        print(type(pairings[0]))
        print(type(data))

        path_pairings = '{}pairings_{}.npy'.format(path_root, name)
        path_data = '{}data_{}.npy'.format(path_root, name)
        path_color = '{}color_{}.npy'.format(path_root, name)
        np.save(path_pairings,pairings[0])
        np.save(path_data, data)
        np.save(path_color, color)

    return data, pairings[0], color




if __name__ == "__main__":
    dataset_sampler = SwissRoll()
    n_samples_array = [128]
    tot_count = len(n_samples_array) * 100
    progress_count = 1
    for n_samples in n_samples_array:
        path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/visualisation_nnsys/final_pretty/vr/'.format(
            n_samples)
        for seed in [30]:
            print('{} out of {}'.format(progress_count, tot_count))
            progress_count += 1
            name = 'vr_ns{}_seed{}'.format(n_samples, seed)

            data, color = dataset_sampler.sample(n_samples, seed = seed)
            data, pairings, color = make_data(data, color, name = name)

            # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            # path_data = '{}data_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            # path_color = '{}color_{}.npy'.format(PATH_ROOT_SWISSROLL, name)
            # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
            #
            make_plot(data, pairings, color, name = name, path_root = path_to_save,cmap = plt.cm.viridis)

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