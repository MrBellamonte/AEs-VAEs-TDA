import random
import time
import uuid

import pandas as pd
import numpy as np

from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":


    unique_id = str(uuid.uuid4())[:4]
    df_timing = pd.DataFrame()

    dataset = SwissRoll()

    n_witnesses = [512,1024,2048]
    #n_landmarks = [int(i) for i in np.logspace(5, 9, num=5, base=2.0)]
    n_landmarks = [8,16, 32, 64]


    data, _ = dataset.sample(n_samples=max(n_witnesses))


    df_timing_data = []

    for n_w in n_witnesses:
        ind_w = random.sample(range(max(n_witnesses)), n_w)
        X_witnesses = data[ind_w,:]

        for n_l in n_landmarks:
            ind_l = random.sample(range(max(n_witnesses)), n_l)
            X_landmarks = data[ind_l, :]
            distance_matrix_not_computed = True
            r_max = 15
            while(distance_matrix_not_computed):
                witness_complex1 = WitnessComplex(X_landmarks ,X_witnesses)
                start = time.time()
                witness_complex1.compute_simplicial_complex(2,r_max=r_max, create_simplex_tree=False, create_metric=True)
                end = time.time()
                if witness_complex1.check_distance_matrix():
                    distance_matrix_not_computed = False
                else:
                    r_max += 2
                    print('Compute again at r_ax={}'.format(r_max))


            time_needed = end-start

            print('nw: {} - nl: {} - time: {}'.format(n_w,n_l,time_needed))

            df_timing_data.append([n_w,n_l,time_needed])




    df_timing = pd.DataFrame(df_timing_data, columns=['n_witnesses', 'n_landmarks', 'time'])

    df_timing.to_csv('/Users/simons/PycharmProjects/MT-VAEs-TDA/output/runtime_approx/witness_complex/timing_worst_{}.csv'.format(unique_id), index=False)