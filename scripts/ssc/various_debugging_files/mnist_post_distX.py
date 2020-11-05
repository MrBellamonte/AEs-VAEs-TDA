from src.data_preprocessing.witness_complex_offline.definitions import (
    NAME_DISTANCES_X_EVAL,
    NAME_DISTANCES_X_TRAIN, NAME_DISTANCES_X_TEST)

import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

root_path = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist'
dataloader_names = ['dataloader_train.pt', 'dataloader_test.pt', 'dataloader_eval.pt']
dataloader_names = ['dataloader_test.pt', 'dataloader_eval.pt']
wcs = ['MNIST_offline-bs64-seed838-noiseNone-20738678',
       'MNIST_offline-bs128-seed838-noiseNone-4f608157',
       'MNIST_offline-bs256-seed838-noiseNone-4a5487de',
       'MNIST_offline-bs512-seed838-noiseNone-ced06774',
       'MNIST_offline-bs1024-seed838-noiseNone-6f31dea2']



name_mapping = {
    'dataloader_train.pt': NAME_DISTANCES_X_TRAIN,
    'dataloader_test.pt' : NAME_DISTANCES_X_TEST,
    'dataloader_eval.pt' : NAME_DISTANCES_X_EVAL
}

for wc in wcs:
    bs = int(wc.split('-')[1][2:])
    for dataloader_name in dataloader_names:

        complete_path = os.path.join(root_path, wc, dataloader_name)

        dataloader = torch.load(complete_path)

        dist_X_all = torch.zeros(len(dataloader), bs, bs)
        for batch_i, (X_batch, label_batch) in enumerate(dataloader):
            dist_X_all[batch_i, :, :] = torch.norm(X_batch[:, None]-X_batch, dim=2, p=2)
        print(dist_X_all.shape)
        torch.save(dist_X_all,
                   os.path.join(root_path, wc, '{}.pt'.format(name_mapping[dataloader_name])))

        print('SUCCESS: {} - {}'.format(wc, dataloader_name))
