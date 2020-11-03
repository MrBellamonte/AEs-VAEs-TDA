import os

import torch
import seaborn as sns
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/scripts/ssc/unity_datasets/scurve_poc'


#sort from position
positions = torch.load(os.path.join(path_to_save,'positions.pt'))
images = torch.load(os.path.join(path_to_save,'images.pt'))

nds = torch.argsort(positions)

print(nds)

indices = (positions-1).numpy()

img_ordered_np = images[nds,:,:].squeeze()

plt.imshow(img_ordered_np[0,:,:])
plt.show()
img_ordered_np = img_ordered_np.view(images.shape[0], -1).numpy()
print(img_ordered_np.shape)
distances = pairwise_distances(img_ordered_np,img_ordered_np,n_jobs=4)

#distances_torch = torch.norm(img_ordered[:, None]-img_ordered, dim=2, p=2)

distances_torch = torch.Tensor(distances)

print(distances_torch.shape)

torch.save(distances_torch, os.path.join(path_to_save,'distances.pt'))

distances_norm = distances
distances_norm = distances_norm/distances_norm.max()

sns.heatmap(distances_norm,cmap='coolwarm',robust = True)
plt.savefig(os.path.join(path_to_save,'heatmap_corgi.pdf'))
plt.show()



