import os

import torch

from src.data_preprocessing.witness_complex_offline.definitions import NAME_DATALOADER_TRAIN
from src.datasets.datasets import Unity_Rotblock, Unity_RotCorgi

# root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/block_rotation_1'
#
# data_loader = torch.load(os.path.join(root_path,'{}.pt'.format(NAME_DATALOADER_TRAIN)))
#
# print(len(data_loader))


dataset = Unity_Rotblock()
print(isinstance(dataset,(Unity_Rotblock,Unity_RotCorgi)))