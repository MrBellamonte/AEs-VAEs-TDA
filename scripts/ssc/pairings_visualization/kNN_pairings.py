import torch
import numpy as np

import matplotlib.pyplot as plt

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

if __name__ == "__main__":
    dataset_sampler = SwissRoll()

    n_samples = 128
    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/visualisation_nnsys/final_pretty/kNN{}'.format(n_samples)

    N_sim = 10

    ks = [1]

    ntot = int(len(ks) * N_sim)
    counter = 1
    for seed in [30]:
        points, color = dataset_sampler.sample(n_samples, seed=seed)
        points_tensor = torch.from_numpy(points)
        pairwise_distances = torch.norm(points_tensor[:, None]-points_tensor, dim=2, p=2)
        sorted, indices = torch.sort(pairwise_distances)

        for k in ks:

            print('{} out of {}'.format(counter, ntot))

            kNN_mask = torch.zeros((points_tensor.size(0), points_tensor.size(0))).scatter(1, indices[:, 1:(k+1)], 1)

            pairings_i = np.where(kNN_mask.numpy() == 1)
            pairings = np.column_stack((pairings_i[0], pairings_i[1]))

            name = 'knn_k{k}_seed{seed}'.format(k = k, seed = seed)

            make_plot(points, pairings, color,name = name, path_root = path_to_save, knn = False, show = True, dpi = 400, cmap = plt.cm.viridis)

            counter += 1





