import os
import random

import torch
from PIL import Image, ImageOps
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.transforms import transforms
from src.data_preprocessing.witness_complex_offline.definitions import *


def angular_metric(ang1, ang2):
    diff = abs(ang1-ang2)
    if diff < 180:
        return diff
    else:
        return (180-diff%180)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

root_path_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_rl'
root_path = '/Users/simons/MT_data/datasets/Unity_simulation/xy_trans_rl'

print(os.listdir(root_path))
transform_to_tensor = transforms.ToTensor()
print('__init__.py' in os.listdir(root_path))

positions = []
images = []

for file in os.listdir(root_path):
    if file == '__init__.py' or '.DS' in file:
        pass
    else:

        pos_temp = torch.ones((3))
        pos_temp[0] = float(file.split('_')[0][1:])/10
        pos_temp[1] = float(file.split('_')[1][1:])/10
        pos_temp[2] = float(file.split('_')[2][3:])
        positions.append(pos_temp)

        pil_img = Image.open(os.path.join(root_path, file)).resize((480, 320)).convert('RGB')
        # transformed = transform_to_tensor(pil_img)
        # pil_img = ImageOps.grayscale(pil_img)
        transformed = transform_to_tensor(pil_img)

        images.append(transformed)

images = torch.stack(images)
positions = torch.stack(positions)
#labels = torch.square(positions[:,:2] - torch.zeros_like(positions[:,:2])).sum(1) # simply the distance from the center.
labels = positions[:,0]
# distances = torch.cdist(images.view(images.shape[0], 3*480*320),
#                         images.view(images.shape[0], 3*480*320))
#
# distances_true = torch.cdist(positions,positions)
#
#
#
torch.save(images,
           os.path.join(root_path_save, '{}.pt'.format('images')))
torch.save(positions,
           os.path.join(root_path_save, '{}.pt'.format('position')))
# torch.save(labels,
#            os.path.join(root_path_save, '{}.pt'.format('labels')))
#
# torch.save(distances,
#            os.path.join(root_path_save, '{}.pt'.format('distances')))
# torch.save(distances_true,
#            os.path.join(root_path_save, '{}.pt'.format('distances_true')))
#
# sns.heatmap((distances/distances.max()),cmap='coolwarm',robust = True)
# plt.savefig(os.path.join(root_path_save,'distances_image.pdf'))
# plt.show()
#
# sns.heatmap((distances_true/distances_true.max()),cmap='coolwarm',robust = True)
# plt.savefig(os.path.join(root_path_save,'distances_true.pdf'))
# plt.show()


full_dataset = TensorDataset(images, labels)
#
# # #ind = random.sample(range(180),180)
# # ind_train = random.sample(range(180),120)
# # ind_test = random.sample(range(180),120)
# # print(positions[ind_train].unsqueeze_(0).shape)
# #
# # train_dataset = TensorDataset(images[ind_train,:,:,:],positions[ind_train])
# # test_dataset = TensorDataset(images[ind_test,:,:,:],positions[ind_test])
torch.save(full_dataset, os.path.join(root_path_save, '{}.pt'.format('full_dataset')))
# # torch.save(train_dataset, os.path.join(root_path_save,'{}.pt'.format('train_dataset')))
# # torch.save(test_dataset, os.path.join(root_path_save,'{}.pt'.format('test_dataset')))
# #
dataloader = DataLoader(full_dataset, batch_size=images.shape[0], pin_memory=True, drop_last=True,
                         shuffle=False)
# # dataloader_eval = DataLoader(train_dataset, batch_size=120, pin_memory=True, drop_last=True,shuffle=False)
# # dataloader_test = DataLoader(test_dataset, batch_size=120, pin_memory=True, drop_last=True,shuffle=False)
# #
torch.save(dataloader, os.path.join(root_path_save, '{}.pt'.format(NAME_DATALOADER_TRAIN)))
# torch.save(dataloader, os.path.join(root_path_save, '{}.pt'.format(NAME_DATALOADER_EVAL)))
# torch.save(dataloader, os.path.join(root_path_save, '{}.pt'.format(NAME_DATALOADER_TEST)))
# #
#
# #
# # for (img,pos) in dataloader_eval:
# #     d_test = pairwise_distances(pos.reshape(-1, 1), pos.reshape(-1, 1),
# #                                  metric=angular_metric)
# #     print('pass test')
# #
# # # save landmark distance matrics
# print(distances.unsqueeze_(0).shape)
# torch.save(distances,
#            os.path.join(root_path_save, '{}.pt'.format(NAME_DISTANCES_TRAIN)))
# torch.save(distances,
#            os.path.join(root_path_save, '{}.pt'.format(NAME_DISTANCES_EVAL)))
# torch.save(distances,
#            os.path.join(root_path_save, '{}.pt'.format(NAME_DISTANCES_TEST)))
