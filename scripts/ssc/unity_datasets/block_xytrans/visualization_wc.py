import os
import random
from functools import reduce

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from src.topology.witness_complex import WitnessComplex

k = 4

root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_l_newpers'
root_path_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_l_newpers/wc_pl'

positions = torch.load(os.path.join(root_path,'position.pt'))
labels = torch.load(os.path.join(root_path,'labels.pt'))
images = torch.load(os.path.join(root_path,'images.pt'))
distances = torch.load(os.path.join(root_path,'landmark_dist_train.pt'))


ind = random.sample(range(400), 200)


positions_b1 = positions[ind,:]
labels_b1 = labels[ind]
images_b1 = images[ind,:,:,:]

X_witnesses = images.view(images.shape[0], reduce((lambda x, y: x*y), images.shape[1:]))
X_batch = images_b1.view(images_b1.shape[0], reduce((lambda x, y: x*y), images_b1.shape[1:]))
witness_complex = WitnessComplex(landmarks=X_batch, witnesses=X_witnesses)
witness_complex.compute_metric_optimized(n_jobs=2)
landmarks_dist_batch = witness_complex.landmarks_dist


torch.save(torch.Tensor(positions_b1),os.path.join(root_path_save,'position.pt'))
torch.save(torch.Tensor(labels_b1),os.path.join(root_path_save,'labels.pt'))
torch.save(torch.Tensor(images_b1),os.path.join(root_path_save,'images.pt'))
torch.save(torch.Tensor(landmarks_dist_batch),os.path.join(root_path_save,'landmark_dist_train.pt'))