
import os
import pickle


import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict

from torchph.pershom import pershom_backend

from src.models.COREL.config import ConfigCOREL, ConfigGrid_COREL


vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1

# config
DEVICE  = "cuda"



def train_TopoAE(data: TensorDataset, config: ConfigCOREL, root_folder, verbose = False):

    pass

def simulator_TopoAE(config_grid: ConfigGrid_COREL, path: str, verbose: bool = False, data_constant: bool = False):

    pass




