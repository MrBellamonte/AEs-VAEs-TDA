from abc import abstractmethod, ABCMeta
from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE
from umap.umap_ import UMAP as UMAP_


class Competitor(metaclass=ABCMeta):
    def __init__(self):
        self.test_eval = False
        pass

    @abstractmethod
    def get_latent_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        pass

    @abstractmethod
    def get_latent_test(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        pass

    @abstractmethod
    def eval(self, **kwargs) -> dict():
        pass


DEFAULT = {
    "tsne"   : dict(),
    "umap"   : dict()
}


class tSNE(TSNE, Competitor):
    """t-SNE"""
    def __init__(self, *args, **kwargs):
        DEFAULT["tsne"].update(kwargs)
        super().__init__(*args, **DEFAULT["tsne"])
        self.test_eval = False

    def get_latent_train(self, x, y):
        return super().fit_transform(x, y=None), y

    def get_latent_test(self, x, y):
        pass

    def eval(self):
        return dict(kl_divergence_  = self.kl_divergence_)


class UMAP(UMAP_, Competitor):
    '''UMAP'''
    def __init__(self, test_eval=True, *args, **kwargs):
        DEFAULT["umap"].update(kwargs)
        super().__init__(*args, **DEFAULT["umap"])
        self.test_eval = test_eval

    def get_latent_train(self, x, y):
        return super().fit_transform(x, y=None), y

    def get_latent_test(self, x, y):
        return super().transform(x), y

    def eval(self):
        return dict()
