from sklearn.metrics import pairwise_distances

from src.datasets.datasets import Spheres

dataset = Spheres()
dataset_l, labels_l = dataset.sample(n_samples=8)

DD = pairwise_distances(dataset_l,dataset_l)

print(DD.max())