from typing import Tuple

import numpy as np
import tadasets
from tadasets import embed

from src.datasets.utils_datagen import label_gen_torus


def dsphere(n: int = 100, d: int = 2, r: float = 1, noise: float = None, ambient: int = None,
            label: int = 0, seed = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Code adopted from TopAE paper (xxx)
    #todo: make proper reference

    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """

    np.random.seed(seed)

    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r*data/np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise*np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data, np.ones(data.shape[0])*label


def torus(n: int = 100, c: float = 2, a: float = 1, noise: float = None, ambient: int = None,
          label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    data = tadasets.torus(n, c, a, noise=noise, ambient=ambient)

    return data, np.ones(data.shape[0])*label



def torus_sectorized(n=800, c=2, a=1, noise=None, ambient=None, seed = 1):
    """
    Sample `n` data points on a torus.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    c : float
        Distance from center to center of tube.
    a : float
        Radius of tube.
    ambient : int, default=None
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    """

    assert a <= c, "That's not a torus"


    # generate seeds for all the generators, to ensure reproducibility even for different number of points sampled
    np.random.seed(seed)
    seeds = np.random.random_integers(0, high=1000, size=3)


    np.random.seed(seeds[0])
    theta = np.random.random((n,)) * 2.0 * np.pi
    np.random.seed(seeds[1])
    phi = np.random.random((n,)) * 2.0 * np.pi

    data = np.zeros((n, 3))
    data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
    data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
    data[:, 2] = a * np.sin(theta)

    if noise:
        np.random.seed(seeds[2])
        data += noise * np.random.randn(*data.shape)

    if ambient:
        data = embed(data, ambient)

    vecfun = np.vectorize(label_gen_torus)

    return data, vecfun(theta, phi)

