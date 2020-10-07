import random
import time

import mnist
import numpy as np

from sklearn.metrics import pairwise_distances

from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":


    #todo: make routine that computes batch-wise witness complex for MNIST and stores in as a dataloader on Euler
    # -> run for multiple seeds... (needed time for bs = 1024; 1000s -> ca. 1.5h needed for entire dataset, with 8 processes....)

    data = '/Users/simons/MT_data/datasets/MNIST/raw_data/imgs'


    #x_train = mnist.train_images()

    X_train = np.vstack([img.reshape(-1,) for img in mnist.train_images()])
    
    ind = random.sample(range(X_train.shape[0]), 1024)

    #X_train_distance = pairwise_distances(X_train[ind,:],X_train, n_jobs=-1)

    start = time.time()
    wc = WitnessComplex(landmarks=X_train[ind,:], witnesses=X_train, n_jobs=4)
    wc.compute_metric_optimized(n_jobs=4)
    end = time.time()

    print('Time needed: {}'.format(end-start))