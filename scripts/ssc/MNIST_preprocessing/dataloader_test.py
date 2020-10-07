import os

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.data_preprocessing.witness_complex_offline.config import ConfigWC
from src.datasets.datasets import SwissRoll

if __name__ == "__main__":

    n_samples = 5
    #
    labels = torch.from_numpy(np.array(range(n_samples)))
    #
    # dataset_train = TensorDataset(labels)
    # dataset_train_2 = TensorDataset(labels)
    #
    # train_loader = DataLoader(dataset_train, batch_size=1, shuffle=False,
    #                           pin_memory=True, drop_last=True)
    #
    # path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/tests'
    # torch.save(labels, os.path.join(path,'labels_tensor.pt'))

    path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/tests'
    os.path.join(path,'labels_tensor.pt')
    labels_loaded = torch.load(os.path.join(path,'labels_tensor.pt'))
    
    print(torch.eq(labels_loaded,labels))

    config = ConfigWC(SwissRoll(), dict(), dict(), 1, 1, 'global_bla', 'root_bla')


    print(config.uid)