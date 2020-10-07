import numpy as np

import matplotlib.pyplot as plt

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

if __name__ == "__main__":
    dataset_sampler = SwissRoll()

    n_samples = 512
    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/visualisation_nnsys/final_pretty/manifold/'.format(n_samples)

    N_sim = 1
    points, color = dataset_sampler.sample(n_samples, seed=1)
    for angle in [0,1,2,3,4,5,6,7,8,9,10]:
        name = 'manifolg_seed{seed}_angle{angle}'.format(seed=1, angle=angle)
        make_plot(points, None, color ,name = name, path_root = path_to_save, knn = False, show = True, dpi = 400, cmap = plt.cm.viridis, angle = angle)
