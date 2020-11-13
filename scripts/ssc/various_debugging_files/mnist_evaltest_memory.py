import time

import torch
from sklearn.metrics import pairwise_distances
from torch import nn

from src.datasets.datasets import MNIST_offline


data_set = MNIST_offline()

data, labels = data_set.sample(train = True)

data_S = data[:1024,:]

print(data.shape)

start1 = time.time()
distance_matrix1 = pairwise_distances(data, data_S, n_jobs = 2)
end1 = time.time()
print('SKLEARN')
print(end1-start1)
print(distance_matrix1.shape)
print(distance_matrix1[1,0])
print(distance_matrix1[0,1])

data_t = torch.from_numpy(data)
data_ts = torch.from_numpy(data_S)
print(data_t.shape)
start1 = time.time()
distance_matrix3 = torch.cdist(data_t, data_ts)

end1 = time.time()
print('TORCH CDIST')
print(end1-start1)
print(distance_matrix3.shape)
print(distance_matrix3[1,0])
print(distance_matrix3[0,1])

distance_matrix1 = torch.from_numpy(distance_matrix1)

err1 = ((distance_matrix1-distance_matrix3)**2).mean()



print('Sklearn - Torch: {}'.format(err1))






