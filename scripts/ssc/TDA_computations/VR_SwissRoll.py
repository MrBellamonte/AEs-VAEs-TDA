import os
import math

import tadasets
from ripser import Rips
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scripts.ssc.visualization.demo_kNN_kwc import annulus, make_scatter
from src.datasets.datasets import SwissRoll
from src.utils.plots import plot_2Dscatter


def computeVR(data, path_to_save):
    rips = Rips()
    diagrams = rips.fit_transform(data)
    rips.plot(diagrams)
    plt.savefig(path_to_save, dpi=200)
    plt.show()
    plt.close()



if __name__ == "__main__":

    path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TDA/SwissRoll'

    #dataset = SwissRoll()
    # small = [0.5,0.8]
    # large = [1, 1.3]
    #
    # area_l = math.pi*(large[1]**2-large[0]**2)
    # area_s = math.pi*(small[1]**2-small[0]**2)
    #
    # sample_ratio = area_l/area_s
    #
    # infty_sign = tadasets.infty_sign(n=3000, noise=0.1)
    for seed in [11,22,33,44]:
        for n in [16,24,32,40,48,56,64,72,80,88,96,112,128]:
            name_plot_vr = 'VR_SwissRoll_n{n}_seed{seed}.pdf'.format(n=n,seed = seed)
            computeVR(data = SwissRoll().sample(n, seed= seed)[0], path_to_save=os.path.join(path, name_plot_vr))

            # df_an1 = annulus(int(sample_ratio*n), large[0], large[1], seed=seed)
    #             # df_an2 = annulus(n, small[0], small[1], label=1, seed=(seed+12))
    #             # df_an = df_an1.append(df_an2, ignore_index=True)
    #             # make_scatter(df_an, name='/annulus_manifold_ns{ns}_nl{nl}_s{s}'.format(ns=n, nl = int(sample_ratio*n), s=seed),
    #             #              base_path=path)
    #             # data = df_an[['x','y']].to_numpy()
    #             #
    #             # name_plot = 'VR_ns{ns}_nl{nl}_seed{s}.pdf'.format(ns=n, nl=int(sample_ratio*n), s=seed)
    #         data = tadasets.infty_sign(n=n, noise=noise)
    #         labels = np.ones(n)
    #         name_plot_m = 'manifold_infty_n{n}_noise{noise}.pdf'.format(n=n, noise=noise)
    #         plot_2Dscatter(data,labels, path_to_save = os.path.join(path, name_plot_m), show = True)
    #         name_plot_vr = 'VR_infty_n{n}_noise{noise}.pdf'.format(n=n, noise=noise)
    #         computeVR(data, path_to_save = os.path.join(path, name_plot_vr))



