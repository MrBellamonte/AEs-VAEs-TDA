import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.transforms import transforms
from src.data_preprocessing.witness_complex_offline.definitions import *

bs = 30
version = 4

def angular_metric(ang1,ang2):
    diff = abs(ang1-ang2)
    if diff < 180:
        return diff
    else:
        return (180-diff%180)


def get_distanes(dataloader,bs):
    d_tensor = torch.zeros((len(dataloader), bs, bs))
    for i,(img, pos) in enumerate(dataloader):
        d_tensor[i,:,:] = torch.tensor(pairwise_distances(pos.reshape(-1, 1), pos.reshape(-1, 1), metric=angular_metric))

    return d_tensor


root_path ='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/corgi_rotation_1/raw_image_data'
root_path_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/corgi_rotation_bw1_l'

print(os.listdir(root_path))
transform_to_tensor = transforms.ToTensor()
print('__init__.py' in os.listdir(root_path))

positions = []
images = []


for file in os.listdir(root_path):
    if file == '__init__.py' :
        pass
    elif file == '.DS_Store' :
        pass
    else:

        deg_pos = int(file.split('_')[0])
        positions.append(deg_pos)

        pil_img = Image.open(os.path.join(root_path,file))
        pil_img = ImageOps.grayscale(pil_img)
        transformed = transform_to_tensor(pil_img)

        images.append(transformed)

images = torch.stack(images)

print(images.shape)
positions = torch.Tensor(positions)

full_dataset = TensorDataset(positions,images)


positions = positions.numpy()
ind_all = random.sample(range(360),360)
deg_landmarks0 = np.linspace(1,359,bs,dtype=int)
deg_landmarks1 =np.linspace(2,360,bs,dtype=int)


ind_l0 = np.in1d(positions, deg_landmarks0).nonzero()[0]
ind_l1= np.in1d(positions, deg_landmarks1).nonzero()[0]
ind_nl = np.array(list(set(ind_all)-set(ind_l0)))
ind_nl = np.array(list(set(ind_nl)-set(ind_l1)))

positions = torch.Tensor(positions)

ind_train_nl = ind_nl[:(360-60-60-2*bs)]
ind_train = np.append(ind_l0,ind_l1)
ind_train = np.append(ind_train,ind_train_nl)
ind_eval = ind_nl[(360-60-60-2*bs):((360-60-60-2*bs)+60)]
ind_test = ind_nl[((360-60-60-2*bs)+60):((360-60-60-2*bs)+2*60)]



train_dataset = TensorDataset(images[ind_train,:,:,:],positions[ind_train])
test_dataset = TensorDataset(images[ind_test,:,:,:],positions[ind_test])
eval_dataset = TensorDataset(images[ind_eval,:,:,:],positions[ind_eval])
torch.save(full_dataset, os.path.join(root_path_save,'{}.pt'.format('full_dataset')))
torch.save(train_dataset, os.path.join(root_path_save,'{}.pt'.format('train_dataset')))
torch.save(test_dataset, os.path.join(root_path_save,'{}.pt'.format('test_dataset')))
torch.save(eval_dataset, os.path.join(root_path_save,'{}.pt'.format('eval_dataset')))

dataloader_train = DataLoader(train_dataset, batch_size=bs, pin_memory=True, drop_last=True,shuffle=False)
dataloader_eval = DataLoader(eval_dataset, batch_size=bs, pin_memory=True, drop_last=True,shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=bs, pin_memory=True, drop_last=True,shuffle=False)

torch.save(dataloader_train, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_TRAIN)))
torch.save(dataloader_eval, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_EVAL)))
torch.save(dataloader_test, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_TEST)))


d_train = get_distanes(dataloader_train,bs)
d_test = get_distanes(dataloader_test,bs)
d_eval = get_distanes(dataloader_eval,bs)


for i, (bs,l) in enumerate(dataloader_train):
    if i == 0:
        print(l)

# save landmark distance matrics
torch.save(d_train,
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_TRAIN)))
torch.save(d_eval,
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_EVAL)))
torch.save(d_test,
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_TEST)))

