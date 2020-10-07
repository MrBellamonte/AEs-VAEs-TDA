import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.ssc.pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":
    dataset_sampler = SwissRoll()

    N_WITNESSES = 2048
    n_samples = 128

    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/visualisation_nnsys/wc{}_w{}/'.format(n_samples,N_WITNESSES)

    N_sim = 100

    ks = [1]
    ntot = int(len(ks) * N_sim)


    counter = 1
    for seed in [30]:
        witnesses, color_ = dataset_sampler.sample((N_WITNESSES-128), seed=seed)
        landmarks, color = dataset_sampler.sample(128, seed=30)

        witnesses_all = np.vstack((witnesses,landmarks))

        witnesses_tensor = torch.from_numpy(witnesses_all)
        landmarks_tensor = torch.from_numpy(landmarks)

        witness_complex = WitnessComplex(landmarks_tensor, witnesses_tensor)
        # witness_complex.compute_simplicial_complex(d_max=1,
        #                                            r_max=10,
        #                                            create_simplex_tree=False,
        #                                            create_metric=True)

        witness_complex.compute_metric_optimized(n_jobs=4)
        for k in ks:

            print('{} out of {}'.format(counter, ntot))

            landmarks_dist = torch.tensor(witness_complex.landmarks_dist)
            sorted, indices = torch.sort(landmarks_dist)
            kNN_mask = torch.zeros((n_samples, n_samples), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)
            pairings_i = np.where(kNN_mask.numpy() == 1)
            pairings = np.column_stack((pairings_i[0], pairings_i[1]))

            name = 'wc{nw}_k{k}_seed{seed}_NEW'.format(nw = N_WITNESSES,k = k, seed = seed)

            make_plot(landmarks, pairings, color,name = name, path_root = path_to_save, knn = False, show = False, dpi = 50,cmap = plt.cm.viridis)

            counter += 1





