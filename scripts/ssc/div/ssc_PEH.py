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

data = np.random.randn(100,2)
data = np.array([[0,0], [1,1],[1,2],[4,2]])
#%%
import torch



vr_simpcomp = ripser(data, maxdim=0)

print(vr_simpcomp['cocycles'])

#%% Try ripser
