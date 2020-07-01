from typing import Tuple

import numpy as np
import tadasets
from tadasets import embed


def dsphere(n: int = 100, d: int = 2, r: float = 1, noise: float = None, ambient: int = None,
            label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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


