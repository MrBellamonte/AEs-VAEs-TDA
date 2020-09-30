import math
import os
import random

from scripts.ssc.visualization.demo_kNN_kwc import annulus
from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":

    path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TDA/SwissRoll'
    dataset = SwissRoll()
    n_w = 512

    #path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TDA/DoubleAnnulus'

    dataset = SwissRoll()
    small = [0.5,0.8]
    large = [1, 1.3]

    area_l = math.pi*(large[1]**2-large[0]**2)
    area_s = math.pi*(small[1]**2-small[0]**2)

    sample_ratio = area_l/area_s

    for seed in [22]:
        for n_l in [64,128]:
            X_w, w_ = dataset.sample(n_w, seed=seed)
            ind = random.sample(range(n_w), n_l)
            X_l, l_ = X_w[ind, :], w_[ind]

            # n_l = int(n_l * (1+sample_ratio))
            #
            # df_an1 = annulus(int(sample_ratio*n_w), large[0], large[1], seed=seed)
            # df_an2 = annulus(n_w, small[0], small[1], label=1, seed=(seed+12))
            # df_an = df_an1.append(df_an2, ignore_index=True)
            #
            # X_w = df_an[['x','y']].to_numpy()
            # ind = random.sample(range(n_w), n_l)
            # X_l = X_w[ind, :]

            witness_complex = WitnessComplex(X_l, X_w)

            witness_complex.compute_simplicial_complex(d_max=2,create_metric=False,create_simplex_tree=True)

            witness_complex.simplex_tree

            name_plot = 'WC_nl{}_nw{}_seed{}.pdf'.format(n_l, n_w, seed)
            witness_complex.get_diagram(show=True, path_to_save=os.path.join(path,name_plot))
