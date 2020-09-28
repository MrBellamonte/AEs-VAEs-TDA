import random
import time

import torch
from sklearn.metrics import pairwise_distances

from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":

    N_WITNESSES = 512
    N_LANDMARKS = 32

    landmark_dist = torch.ones(N_LANDMARKS, N_LANDMARKS) * 1000000

    dataset = SwissRoll()
    X_witnesses, _ = dataset.sample(n_samples=N_WITNESSES)

    ind_l = random.sample(range(N_WITNESSES), N_LANDMARKS)
    X_landmarks = X_witnesses[ind_l, :]

    witness_complex1 = WitnessComplex(X_landmarks, X_witnesses)

    for n_jobs in [1]:
        witness_complex1 = WitnessComplex(X_landmarks, X_witnesses)
        start = time.time()
        witness_complex1.compute_metric_optimized(n_jobs=-1)
        end = time.time()
        print('{} jobs --- Time needed: {}'.format(n_jobs, end-start))

        witness_complex2 = WitnessComplex(X_landmarks, X_witnesses)
        start = time.time()
        witness_complex2.compute_metric_optimized(n_jobs=8)
        end = time.time()
        print('{} jobs --- Time needed: {}'.format(n_jobs, end-start))

        witness_complex3 = WitnessComplex(X_landmarks, X_witnesses)
        start = time.time()
        witness_complex3.compute_simplicial_complex(d_max=1, create_metric = True)
        end = time.time()
        print('{} jobs --- Time needed: {}'.format(n_jobs, end-start))


        print('1 job')
        print(witness_complex1.landmarks_dist)

        print('2 job')
        print(witness_complex2.landmarks_dist)

        print('old')
        print(witness_complex3.landmarks_dist)


