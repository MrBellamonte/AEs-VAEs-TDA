"""Base class for autoencoder models."""
import abc
from typing import Dict, Tuple, List, Any

from torch import Tensor, nn



class VariationalAutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for variational autoencoders."""

    @abc.abstractmethod
    def forward(self, x) -> Tuple[float, Dict[str, float]]:
        """Compute loss for model.

        Args:
            x: Tensor with data

        Returns:
            Tuple[loss, dict(loss_component_name -> loss_component)]

        """

    @abc.abstractmethod
    def sample(self,batch_size):
        """:arg"""

    @abc.abstractmethod
    def encode(self, x):
        """Compute latent representation."""

    @abc.abstractmethod
    def decode(self, z):
        """Compute reconstruction."""

