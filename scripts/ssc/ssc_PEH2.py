import numpy as np
import torch
from torch.utils.data import TensorDataset
from ripser import ripser
from src.datasets.datasets import Spheres
from src.models.COREL.eval_engine import get_model, get_latentspace_representation

from src.datasets.datasets import Spheres
from src.models.COREL.eval_engine import get_model

data = np.random.randn(5,2)
data = np.array([[0,0], [1,1],[1,2],[4,2]])
#%%
import torch

data_tensor = torch.Tensor(data)

#%%
from torchph.pershom import pershom_backend

# config
DEVICE  = "cuda"

data_tensor = data_tensor.to(DEVICE)

d0_PEH = pershom_backend.vr_persistence_l1(data_tensor,0,0)
#%%
print(d0_PEH)
#
d0_PEH1 = pershom_backend.vr_h0_pairings_l1(data_tensor,0,0)
print(d0_PEH1)