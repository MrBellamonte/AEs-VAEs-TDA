import os

import torch
import seaborn as sns
import matplotlib.pyplot as plt

k = 4
os.environ['KMP_DUPLICATE_LIB_OK']='True'
root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_l_newpers/wc_pl'

positions = torch.load(os.path.join(root_path,'position.pt'))
labels = torch.load(os.path.join(root_path,'labels.pt'))
images = torch.load(os.path.join(root_path,'images.pt'))
distances = torch.load(os.path.join(root_path,'landmark_dist_train.pt'))

distances = torch.cdist(images.view(images.shape[0], 3*480*320),
                        images.view(images.shape[0], 3*480*320))


sorted, indices = torch.sort(distances[:,:])
kNN_mask = torch.zeros((distances.shape[1], distances.shape[1]), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)

positions = positions.numpy()
sns.scatterplot(positions[:,0], positions[:,1], hue = labels.numpy())

for i in range(distances.shape[1]):
    j = indices[i,1]
    plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    if k >= 2:
        j = indices[i,2]
        plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    if k >= 3:
        j = indices[i,3]
        plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    if k >= 4:
        j = indices[i,4]
        plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    if k >= 5:
        j = indices[i,5]
        plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    if k >= 6:
        j = indices[i,6]
        plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    # j = indices[i,1]
    # plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    # j = indices[i,1]
    # plt.plot([positions[i, 0],positions[j, 0]],[positions[i, 1],positions[j, 1]], color='grey',zorder=5)
    #
    #

plt.show()
