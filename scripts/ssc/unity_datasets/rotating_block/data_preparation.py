import os
import random

import torch
from PIL import Image
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.transforms import transforms
from src.data_preprocessing.witness_complex_offline.definitions import *



def angular_metric(ang1,ang2):
    diff = abs(ang1-ang2)
    if diff < 180:
        return diff
    else:
        return (180-diff%180)


root_path ='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/block_rotation_1/raw_image_data'
root_path_save ='/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/block_rotation_1'

print(os.listdir(root_path))
transform_to_tensor = transforms.ToTensor()
print('__init__.py' in os.listdir(root_path))


positions = []
images = []


for file in os.listdir(root_path):
    if file == '__init__.py' :
        pass 
    else:

        deg_pos = int(file.split('_')[0])
        positions.append(deg_pos)

        pil_img = Image.open(os.path.join(root_path,file)).convert('RGB')
        transformed = transform_to_tensor(pil_img)

        images.append(transformed)

images = torch.stack(images)
positions = torch.Tensor(positions)

full_dataset = TensorDataset(positions,images)

#ind = random.sample(range(180),180)
ind_train = random.sample(range(180),120)
ind_test = random.sample(range(180),120)
print(positions[ind_train].unsqueeze_(0).shape)

train_dataset = TensorDataset(images[ind_train,:,:,:],positions[ind_train])
test_dataset = TensorDataset(images[ind_test,:,:,:],positions[ind_test])
torch.save(full_dataset, os.path.join(root_path_save,'{}.pt'.format('full_dataset')))
torch.save(train_dataset, os.path.join(root_path_save,'{}.pt'.format('train_dataset')))
torch.save(test_dataset, os.path.join(root_path_save,'{}.pt'.format('test_dataset')))

dataloader_train = DataLoader(train_dataset, batch_size=120, pin_memory=True, drop_last=True,shuffle=False)
dataloader_eval = DataLoader(train_dataset, batch_size=120, pin_memory=True, drop_last=True,shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=120, pin_memory=True, drop_last=True,shuffle=False)

torch.save(dataloader_train, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_TRAIN)))
torch.save(dataloader_eval, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_EVAL)))
torch.save(dataloader_test, os.path.join(root_path_save,'{}.pt'.format(NAME_DATALOADER_TEST)))

for (img,pos) in dataloader_train:
    d_train = pairwise_distances(pos.reshape(-1, 1), pos.reshape(-1, 1),metric=angular_metric)
    print('pass train')

for (img,pos) in dataloader_eval:
    d_test = pairwise_distances(pos.reshape(-1, 1), pos.reshape(-1, 1),
                                 metric=angular_metric)
    print('pass test')

# save landmark distance matrics
torch.save(torch.Tensor(d_train).unsqueeze_(0),
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_TRAIN)))
torch.save(torch.Tensor(d_train).unsqueeze_(0),
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_EVAL)))
torch.save(torch.Tensor(d_test).unsqueeze_(0),
           os.path.join(root_path_save,'{}.pt'.format(NAME_DISTANCES_TEST)))