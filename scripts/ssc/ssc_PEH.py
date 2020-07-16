# Generate random points in 2D plane
# Use pershom package to calculate d-dim perh
# Plot

#%% Genreate 2D data

import numpy as np
import torch
from torch.utils.data import TensorDataset
from ripser import ripser
from src.datasets.datasets import Spheres
from src.models.COREL.eval_engine import get_model, get_latentspace_representation

from src.datasets.datasets import Spheres
from src.models.COREL.eval_engine import get_model

data = np.random.randn(100,100)

#%%
import torch

data_tensor = torch.Tensor(data)

#%%
from torchph.pershom import pershom_backend
vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1
# config
DEVICE  = "cuda"

data_tensor = data_tensor.to(DEVICE) 

d0_PEH = pershom_backend.vr_persistence_l1(data_tensor,0,0)

#%%

DATASET_ = Spheres()
X, y = DATASET_.sample(n_samples=500)

dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))

path_source = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/tshinge/2020-07-05/Spheres-d100-n_spheres11-r5-n_samples500-autoencoder-128-64-32-lr1_1000-bs64-nep40-rlL1Loss-reductionmean-rlw1-tlTwoSidedHingeLoss-reductionmean-ratio1_4-penalty_typesquared-tlw8-8f4d4099/'


model = get_model(path_source, config_fix=True)
X, Y, Z = get_latentspace_representation(model, dataset)

vr_simpcomp = ripser(Z, maxdim=0)



#%% Try ripser

print(vr_simpcomp['dgms'][0][:,1][:-1])

print(vr_simpcomp['dgms'][0][:,1][-1])


#%% Plot!
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

sns.violinplot(data=vr_simpcomp['dgms'][0][:,1][:-1], palette="Set3", bw=.2, cut=1, linewidth=1)
plt.show()


#%%sns.set(style="whitegrid")

sns.boxplot(data=vr_simpcomp['dgms'][0][:,1][:-1], palette="Set3", linewidth=1)
plt.show()