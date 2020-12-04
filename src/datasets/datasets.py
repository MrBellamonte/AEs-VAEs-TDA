import os
import random
from abc import ABCMeta, abstractmethod
from typing import Tuple

import mnist
import numpy as np
import sklearn
import torch
from sklearn import datasets

from .shapes import dsphere, torus


class DataSet(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n_samples: int, seed: int, noise: float, train: bool) -> Tuple[
        np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def sample_manifold(self, n_samples: int, seed: int, noise: float, train: bool) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def fancy_name(self) -> str:
        pass


DEFAULT = {
    "spheres"      : dict(d=100, n_spheres=11, r=5, seed=42, noise=0, ratio_largesphere=10),
    "doubletorus"  : dict(c1=6, a1=4, c2=6, a2=1, seed=1, noise=0),
    "swissroll"    : dict(seed=1, noise=0),
    "mnist"        : dict(n_samples=1000000, seed=None,normalization=True),
    "mnist_offline": dict(n_samples=1000000, seed=None,normalization=True,
                          root_path='/Users/simons/PycharmProjects/MT-VAEs-TDA'),
    "unity_rotating_block": dict(root_path='/Users/simons/PycharmProjects/MT-VAEs-TDA'),
    "unity_rotating_corgi": dict(root_path='/Users/simons/PycharmProjects/MT-VAEs-TDA'),
    "unity_xytrans": dict(root_path='/Users/simons/PycharmProjects/MT-VAEs-TDA', version = 'xy_trans'),
}


class Spheres(DataSet):
    '''
    Spheres dataset from TopAE paper
    #todo: make proper reference
    '''

    fancy_name = "Spheres dataset"
    __slots__ = ['d', 'n_spheres', 'r']

    def __init__(self, d=DEFAULT['spheres']['d'], n_spheres=DEFAULT['spheres']['n_spheres'],
                 r=DEFAULT['spheres']['r']):
        self.d = d
        self.n_spheres = n_spheres
        self.r = r

    def sample(self, n_samples, noise=0, seed=DEFAULT['spheres']['seed'],
               ratio_largesphere=DEFAULT['spheres']['ratio_largesphere'], train: bool = True,
               old_implementation=False):
        # todo Parametrize ratio of samples between small and big spheres
        np.random.seed(seed)
        if old_implementation:
            # Deprecated! Don't use, except for evaluating old experiments.
            seeds = np.random.random_integers(0, high=1000, size=self.n_spheres)
        else:
            seeds = np.random.randint(0, high=1000, size=(2, self.n_spheres))
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
            sphere, labels = dsphere(n=n_samples, d=self.d, r=self.r, seed=seeds[i], noise=noise)
            spheres.append(sphere+shift_matrix[i, :])
            n_datapoints += n_samples

        # Additional big surrounding sphere:
        n_samples_big = ratio_largesphere*n_samples
        big, labels = dsphere(n=n_samples_big, d=self.d, r=self.r*5, seed=seeds[-1], noise=noise)
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

    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))


class DoubleTorus(DataSet):
    '''
    Double Torus
    '''
    fancy_name = "Double torus dataset"
    __slots__ = ['c1', 'a1', 'c2', 'a2']

    def __init__(self, c1=DEFAULT['doubletorus']['c1'], a1=DEFAULT['doubletorus']['a1'],
                 c2=DEFAULT['doubletorus']['c2'], a2=DEFAULT['doubletorus']['a2']):
        self.c1 = c1
        self.a1 = a1
        self.c2 = c2
        self.a2 = a2

    def sample(self, n_samples, noise=DEFAULT['doubletorus']['noise'],
               seed=DEFAULT['doubletorus']['seed'], train=True):
        if train:
            pass
        else:
            print('Test mode not implemented for DoubleTorus')
        # todo: Implement "manually" s.t. seed selection possible.

        # outer torus
        data1, labels1 = torus(n=n_samples, c=6, a=4, label=0, noise=noise)
        # inner torus
        data2, labels2 = torus(n=n_samples, c=6, a=1, label=1, noise=noise)

        return np.concatenate((data1, data2), axis=0), np.concatenate((labels1, labels2), axis=0)

    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))


class SwissRoll(DataSet):
    '''
    Swiss roll
    '''
    fancy_name = "Swiss Roll Dataset"

    __slots__ = []

    def __init__(self):
        pass

    def sample(self, n_samples, noise=DEFAULT['swissroll']['noise'],
               seed=DEFAULT['swissroll']['seed'], train=True):
        np.random.seed(seed=seed)
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
        else:
            seed = seeds[1]

        generator = sklearn.utils.check_random_state(seed)
        t = 1.5*np.pi*(1+2*generator.rand(1, n_samples))
        x = t*np.cos(t)
        y = 21*generator.rand(1, n_samples)
        z = t*np.sin(t)

        X_transformed = np.concatenate((x, y, z))
        X_transformed += noise*generator.randn(3, n_samples)
        X_transformed = X_transformed.T

        t = np.squeeze(t)

        return X_transformed, t

    def sample_manifold(self, n_samples, noise=DEFAULT['swissroll']['noise'],
                        seed=DEFAULT['swissroll']['seed'], train=True):
        np.random.seed(seed=seed)
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
        else:
            seed = seeds[1]

        generator = sklearn.utils.check_random_state(seed)
        t = 1.5*np.pi*(1+2*generator.rand(1, n_samples))
        x = t*np.cos(t)
        y = 21*generator.rand(1, n_samples)
        z = t*np.sin(t)

        X_transformed = np.concatenate((x, y, z))
        X_transformed += noise*generator.randn(3, n_samples)
        X_transformed = X_transformed.T

        X_manifold = np.concatenate((t, y))
        X_manifold = X_manifold.T
        t = np.squeeze(t)

        return X_manifold, X_transformed, t

    def sample_old(self, n_samples, noise=DEFAULT['swissroll']['noise'],
                   seed=DEFAULT['swissroll']['seed'], train=True):
        # todo change to fixed increment for test?
        np.random.seed(seed=seed)
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
        else:
            seed = seeds[1]

        return datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)


class MNIST(DataSet):
    __slots__ = []
    fancy_name = "MNIST Dataset"

    def __init__(self):
        pass

    def sample(self, n_samples=DEFAULT['mnist']['n_samples'], seed=DEFAULT['mnist']['seed'], normalization=DEFAULT['mnist']['normalization'],
               train=True):

        if seed is None:
            seeds = [0, 1]
        else:
            np.random.seed(seed=seed)
            seeds = np.random.randint(0, high=1000, size=2)

        if (n_samples is DEFAULT['mnist']['n_samples']) and (seed is not None):
            print('USER WARNING: Seed is ignored.')

        if train:
            seed = seeds[0]
            data = np.vstack([img.reshape(-1, ) for img in mnist.train_images()])
            if normalization:
                data = data/255
            labels = mnist.train_labels()
        else:
            seed = seeds[1]
            data = np.vstack([img.reshape(-1, ) for img in mnist.test_images()])
            if normalization:
                data = data/255
            labels = mnist.test_labels()

        random.seed(seed)
        if (n_samples is None) or (n_samples >= data.shape[0]):
            return data, labels
        else:
            ind = random.sample(range(data.shape[0]), n_samples)
            return data[ind, :], labels[ind, :]

    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))


class MNIST_offline(DataSet):
    __slots__ = []
    fancy_name = "MNIST Dataset Offline"

    def __init__(self):
        pass

    def sample(self, n_samples=DEFAULT['mnist_offline']['n_samples'],
               seed=DEFAULT['mnist_offline']['seed'], train=True,
               root_path=DEFAULT['mnist_offline']['root_path'],
               normalization=DEFAULT['mnist']['normalization']):

        if seed is None:
            seeds = [0, 1]
        else:
            np.random.seed(seed=seed)
            seeds = np.random.randint(0, high=1000, size=2)

        if (n_samples is DEFAULT['mnist']['n_samples']) and (seed is not None):
            print('USER WARNING: Seed is ignored.')

        if train:
            seed = seeds[0]
            data = np.load(os.path.join(root_path, 'src/datasets/mnist_data/train_data.npy'))

            #avoid that data gets normalized twice
            if (data.max()>200) and normalization:
                data = data/255
            labels = np.load(os.path.join(root_path, 'src/datasets/mnist_data/train_labels.npy'))
        else:
            seed = seeds[1]
            data = np.load(os.path.join(root_path, 'src/datasets/mnist_data/test_data.npy'))
            # avoid that data gets normalized twice
            if (data.max()>200) and normalization:
                data = data/255
            labels = np.load(os.path.join(root_path, 'src/datasets/mnist_data/test_labels.npy'))

        random.seed(seed)
        if (n_samples is None) or (n_samples >= data.shape[0]):
            ind = random.sample(range(data.shape[0]), data.shape[0])
            return data[ind, :], labels[ind]
        else:
            ind = random.sample(range(data.shape[0]), n_samples)
            return data[ind, :], labels[ind]

    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))


class Unity_Rotblock(DataSet):
    __slots__ = []
    fancy_name = "Unity rotating block"

    def __init__(self):
        pass

    def sample(self, train = True,root_path=DEFAULT['unity_rotating_block']['root_path'], seed = None):

        if train:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/block_rotation_1/train_dataset.pt'))

        else:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/block_rotation_1/test_dataset.pt'))

        position = data[:][:][1].numpy()
        images = data[:][:][0].numpy()

        return images, position


    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))


class Unity_RotCorgi(DataSet):
    __slots__ = []
    fancy_name = "Unity rotating corgi"

    def __init__(self):
        pass

    def sample(self,landmarks = False, version=1 , train = True, root_path=DEFAULT['unity_rotating_corgi']['root_path'], seed = None):

        if landmarks:
            suffix = '_l'
        else:
            suffix = ''

        if train:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/corgi_rotation_{}{}/train_dataset.pt'.format(version,suffix)))

        else:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/corgi_rotation_{}{}/test_dataset.pt'.format(version,suffix)))

        position = data[:][:][1].numpy()
        images = data[:][:][0].numpy()

        return images, position


    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))

class Unity_RotOpenAI(DataSet):
    __slots__ = []
    fancy_name = "Unity rotating block"

    def __init__(self):
        pass

    def sample(self, train = True,root_path=DEFAULT['unity_rotating_block']['root_path'], seed = None):

        if train:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/openai_rotating/full_dataset.pt'))

        else:
            data = torch.load(os.path.join(root_path, 'src/datasets/simulated/openai_rotating/full_dataset.pt'))

        position = data[:][:][1].numpy()
        images = data[:][:][0].numpy()

        return images, position


    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))

class Unity_XYTransOpenAI(DataSet):
    __slots__ = ['version']
    fancy_name = "Unity rotating block"

    def __init__(self, version = DEFAULT['unity_xytrans']['version']):

        assert version in ['xy_trans', 'xy_trans_l', 'xy_trans_l_newpers', 'xy_trans_rot','xy_trans_final']

        self.version = version

    def sample(self, train = True,root_path=DEFAULT['unity_xytrans']['root_path'], seed = None):

        if train:
            data = torch.load(
                os.path.join(root_path, 'src/datasets/simulated/{}/full_dataset.pt'.format(self.version)))

        else:
            data = torch.load(
                os.path.join(root_path, 'src/datasets/simulated/{}/full_dataset.pt'.format(self.version)))

        position = data[:][:][1].numpy()
        images = data[:][:][0].numpy()

        return images, position


    def sample_manifold(self):
        raise AttributeError('{} cannot sample from manifold.'.format(self.fancy_name))