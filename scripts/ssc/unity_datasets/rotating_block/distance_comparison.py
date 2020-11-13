import os

import torch
import seaborn as sns
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/scripts/ssc/unity_datasets/rotating_block'
root_path_data = '/Users/simons/PycharmProjects/MT-VAEs-TDA/scripts/ssc/unity_datasets/rotating_block/data_blockopenai'

# #sort from position
# images = torch.load(os.path.join(root_path_data,'{}.pt'.format('images')))
# positions = torch.load(os.path.join(root_path_data,'{}.pt'.format('deg_pos')))
#
# nds = torch.argsort(positions)
#
# print(nds)
#
# indices = (positions-1).numpy()
#
# img_ordered_np = images[nds,:,:].squeeze()
#
# plt.imshow(img_ordered_np[0,:,:])
# plt.show()
# img_ordered_np = img_ordered_np.view(images.shape[0], -1).numpy()
# print(img_ordered_np.shape)
# distances = pairwise_distances(img_ordered_np,img_ordered_np,n_jobs=4)
#
# #distances_torch = torch.norm(img_ordered[:, None]-img_ordered, dim=2, p=2)
#
# distances_torch = torch.Tensor(distances)
#
# print(distances_torch.shape)
#
# torch.save(distances_torch, os.path.join(path_to_save,'distances.pt'))

distances_norm = torch.load(os.path.join(path_to_save,'distances.pt')).numpy()
distances_norm = distances_norm/distances_norm.max()

sns.heatmap(distances_norm,cmap='coolwarm',robust = True)
plt.savefig(os.path.join(path_to_save,'heatmap_block.pdf'))
plt.show()

ks = [1,2,3,4,5,6]
for k in ks:
    dist_temp = torch.from_numpy(distances_norm)
    sorted, indices = torch.sort(dist_temp)
    kNN_mask = torch.zeros((distances_norm.shape[0], distances_norm.shape[0]), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)


    sns.heatmap(kNN_mask.numpy(),cmap='coolwarm',robust = True)
    plt.savefig(os.path.join(path_to_save,'heatmap_block_{}nn.pdf'.format(k)))
    plt.show()

