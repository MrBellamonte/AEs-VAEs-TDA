import sys

import gudhi
import time
from guppy import hpy
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from scripts.ssc.visualization.demo_kNN_kwc import annulus
from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex





if __name__ == "__main__":

    n_landmarks = 128
    n_witnesses = 2048
    seed = 0

    df_l = annulus(n_landmarks,1,1.25, seed = 1)
    X_landmarks = df_l[['x', 'y']].to_numpy()
    df_w = annulus(n_witnesses,1,1.25, seed = 2)
    X_witnesses = df_w[['x', 'y']].to_numpy()

    witness_complex1 = WitnessComplex(X_landmarks,X_witnesses)
    start = time.time()
    witness_complex1.compute_simplicial_complex(2, r_max=1, create_simplex_tree=False, create_metric=True)
    end = time.time()
    print('Time needed single core: {}'.format(end-start))


    # witness_complex1new = WitnessComplex(X_landmarks,X_witnesses, new = True)
    # start = time.time()
    # witness_complex1new.compute_simplicial_complex(2, r_max=10, create_simplex_tree=False, create_metric=True)
    # end = time.time()
    # print('Time needed single core: {}'.format(end-start))

    # witness_complex1new = WitnessComplex(X_landmarks,X_witnesses, new = True)
    # start = time.time()
    # witness_complex1new.compute_simplicial_complex(1, r_max=2, create_simplex_tree=False, create_metric=True)
    # end = time.time()
    # print('Time needed single core: {}'.format(end-start))
    # print(witness_complex1new.landmarks_dist)



    # for n in [4]:
    #     witness_complex2 = WitnessComplex(X_landmarks, X_witnesses)
    #     n_jobs = n+1
    #     start = time.time()
    #
    #     witness_complex2.compute_simplicial_complex_parallel(1,r_max=2, create_simplex_tree=False,create_metric=True, n_jobs = n_jobs)
    #     end = time.time()
    #     print('Time needed multi core: {time}    ({n_jobs} cores)'.format(time = (end-start), n_jobs = n_jobs))
    #     #print((witness_complex1new.landmarks_dist-witness_complex2.landmarks_dist).sum())



    #witness_complex.get_diagram(show = True)





