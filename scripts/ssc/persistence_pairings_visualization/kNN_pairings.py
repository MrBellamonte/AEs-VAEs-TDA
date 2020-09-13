import torch
import numpy as np

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

if __name__ == "__main__":
    dataset_sampler = SwissRoll()

    n_samples = 2560
    seed = 534
    k = 80

    points, color = dataset_sampler.sample(n_samples, seed=seed)

    points_tensor = torch.from_numpy(points)

    pairwise_distances = torch.norm(points_tensor[:, None]-points_tensor, dim=2, p=2)

    sorted, indices = torch.sort(pairwise_distances)
    kNN_mask = torch.zeros((points_tensor.size(0), points_tensor.size(0))).scatter(1, indices[:, 1:(k+1)], 1)

    pairings_i = indices = np.where(kNN_mask.numpy() == 1)
    pairings = np.column_stack((pairings_i[0], pairings_i[1]))


    print('pairings computed!')
    make_plot(points, pairings, color, path_root=None)

