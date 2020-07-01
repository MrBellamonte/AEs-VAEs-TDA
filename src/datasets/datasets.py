from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .shapes import dsphere, torus


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    '''
    Code adopted from TopAE paper (xxx)
    #todo: make proper reference
    '''
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere, labels= dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere+shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10*n_samples  # int(n_samples/2)
    big, labels = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index+n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels


def double_tours(n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    #todo allow furhter parametrization
    data1, labels1 = torus(n=n_samples, c=6, a=4, label=0) # outer torus
    data2, labels2 = torus(n=n_samples, c=6, a=1, label=1) # inner torus
    return np.vstack((data1, data2)), np.vstack((labels1, labels2))
