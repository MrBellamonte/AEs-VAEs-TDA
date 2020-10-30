import os

import torch
from sklearn.metrics import pairwise_distances

root_path_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/corgi_rotation_bw1_l'

dataloader_path = 'dataloader_train.pt'
distances_save = 'distances_bs0.pt'

dataloader = torch.load(os.path.join(root_path_save,dataloader_path))

for i, (bs,label) in enumerate(dataloader):

    if i > 0:
        break

    # compute pairwise distance matrix SAVE!
    distances = torch.tensor(pairwise_distances(bs.reshape(-1, 1), bs.reshape(-1, 1)))
    print(distances.shape)
    torch.save(os.path.join(root_path_save, distances_save))
    # compute kNN mask
    # compare kNN mask