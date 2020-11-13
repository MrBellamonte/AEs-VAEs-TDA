import random
import time

import mnist
import numpy as np
import torch

from sklearn.metrics import pairwise_distances
from torch.utils.data import TensorDataset

from src.datasets.datasets import MNIST, MNIST_offline
from src.topology.witness_complex import WitnessComplex

if __name__ == "__main__":


    #todo: make routine that computes batch-wise witness complex for MNIST and stores in as a dataloader on Euler
    # -> run for multiple seeds... (needed time for bs = 1024; 1000s -> ca. 1.5h needed for entire dataset, with 8 processes....)



    #x_train = mnist.train_images()

    dataset = MNIST_offline()




    data, labels = dataset.sample(train = False)




    # np.save('/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/mnist_data/test_data.npy', data)
    #np.save('/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/mnist_data/test_labels.npy',labels)



    print(data.shape)
    print(labels.shape)
    # ind = random.sample(range(60000),4096)
    #
    #
    # start = time.time()
    # wc = WitnessComplex(landmarks=data[ind,:], witnesses=data)
    # wc.compute_metric_optimized(n_jobs=4)
    # end = time.time()
    # #
    # print('Time needed: {}'.format(end-start))