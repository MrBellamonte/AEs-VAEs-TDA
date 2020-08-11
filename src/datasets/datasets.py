from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

from .shapes import dsphere, torus



class DataSet(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n_samples: int, seed: int, noise: float, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def fancy_name(self) -> str:
        pass


DEFAULT = {
    "spheres"   : dict(d = 100, n_spheres = 11, r = 5, seed = 42, noise = 0, ratio_largesphere = 10),
    "doubletorus" : dict(c1 = 6, a1 = 4, c2 = 6, a2 = 1, seed = 1, noise = 0),
    "swissroll" : dict(seed = 1, noise = 0)
}


class Spheres(DataSet):
    '''
    Spheres dataset from TopAE paper
    #todo: make proper reference
    '''

    fancy_name = "Spheres dataset"
    __slots__ = ['d', 'n_spheres', 'r']

    def __init__(self, d=DEFAULT['spheres']['d'], n_spheres=DEFAULT['spheres']['n_spheres'], r=DEFAULT['spheres']['r']):
        self.d = d
        self.n_spheres = n_spheres
        self.r = r

    def sample(self, n_samples, noise = 0,seed = DEFAULT['spheres']['seed'], ratio_largesphere = DEFAULT['spheres']['ratio_largesphere'], train: bool = True, old_implementation = False):
        #todo Parametrize ratio of samples between small and big spheres
        np.random.seed(seed)
        if old_implementation:
            #Deprecated! Don't use, except for evaluating old experiments.
            seeds = np.random.random_integers(0, high=1000, size=self.n_spheres)
        else:
            seeds = np.random.randint(0, high=1000, size=(2,self.n_spheres))
            if train:
                seeds = seeds[0][:]
            else:
                seeds = seeds[1][:]

        # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
        # Fixed seed for shift matrix!
        np.random.seed(42)
        variance = 10/np.sqrt(self.d)

        shift_matrix = np.random.normal(0, variance, [self.n_spheres, self.d+1])

        spheres = []
        n_datapoints = 0
        for i in np.arange(self.n_spheres-1):
            sphere, labels = dsphere(n=n_samples, d=self.d, r=self.r, seed = seeds[i], noise = noise)
            spheres.append(sphere+shift_matrix[i, :])
            n_datapoints += n_samples

        # Additional big surrounding sphere:
        n_samples_big = ratio_largesphere*n_samples
        big, labels = dsphere(n=n_samples_big, d=self.d, r=self.r*5, seed = seeds[-1],noise = noise)
        spheres.append(big)
        n_datapoints += n_samples_big


        # Create Dataset:
        dataset = np.concatenate(spheres, axis=0)

        labels = np.zeros(n_datapoints)
        label_index = 0
        for index, data in enumerate(spheres):
            n_sphere_samples = data.shape[0]
            labels[label_index:label_index+n_sphere_samples] = index
            label_index += n_sphere_samples

        return dataset, labels


class DoubleTorus(DataSet):
    '''
    Double Torus
    '''
    fancy_name = "Double torus dataset"
    __slots__ = ['c1', 'a1','c2', 'a2']

    def __init__(self, c1=DEFAULT['doubletorus']['c1'], a1=DEFAULT['doubletorus']['a1'], c2 = DEFAULT['doubletorus']['c2'], a2=DEFAULT['doubletorus']['a2']):
        self.c1 = c1
        self.a1 = a1
        self.c2 = c2
        self.a2 = a2

    def sample(self, n_samples, noise = DEFAULT['doubletorus']['noise'], seed = DEFAULT['doubletorus']['seed'], train = True):
        if train:
            pass
        else:
            print('Test mode not implemented for DoubleTorus')
        #todo: Implement "manually" s.t. seed selection possible.

        # outer torus
        data1, labels1 = torus(n=n_samples, c=6, a=4, label=0, noise=noise)
        # inner torus
        data2, labels2 = torus(n=n_samples, c=6, a=1, label=1, noise=noise)

        return np.concatenate((data1, data2), axis=0), np.concatenate((labels1, labels2), axis=0)


class SwissRoll(DataSet):
    '''
    Swiss roll
    '''
    fancy_name = "Swiss Roll Dataset"

    __slots__ = []
    def __init__(self):
        pass

    def sample(self, n_samples, noise = DEFAULT['swissroll']['noise'], seed = DEFAULT['swissroll']['seed'], train = True):
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
        else:
            seed = seeds[1]

        return datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)

