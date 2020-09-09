"""Topolologically regularized autoencoder using approximation."""
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from src.models.autoencoder.base import AutoencoderModel


class TopologicallyRegularizedAutoencoderWC(AutoencoderModel):
    """Topologically regularized autoencoder."""

    def __init__(self,autoencoder,lam_t=1.,lam_r=1., toposig_kwargs=None):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.lam_t = lam_t
        self.lam_r = lam_r
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.k = toposig_kwargs['k']
        self.topo_sig = TopologicalSignatureDistanceWC(**toposig_kwargs)
        self.autoencoder = autoencoder
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x, dist_X, pair_mask_X):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """

        dist_X = torch.norm(x[:, None]-x, dim=2, p=2)
        #todo check if this normlization is sensible...
        dist_X = dist_X / dist_X.max()


        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)
        latent = self.autoencoder.encode(x)
        topo_error, topo_error_components = self.topo_sig(latent,self.latent_norm, dist_X, pair_mask_X)


        # normalize topo_error according to batch_size
        batch_size = x.size(0)
        topo_error = topo_error / (float(batch_size)*self.k)
        loss = self.lam_r * ae_loss + self.lam_t * topo_error
        loss_components = {
            'loss.autoencoder': self.lam_r * ae_loss,
            'loss.topo_error': self.lam_t * topo_error
        }
        loss_components.update(topo_error_components)
        loss_components.update(ae_loss_comp)
        return (
            loss,
            loss_components
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


class TopologicalSignatureDistanceWC(nn.Module):
    """Topological signature."""

    def __init__(self, k, match_edges):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.k = k
        self.match_edges = match_edges



    def _get_pairings_dist_Z(self, latent,latent_norm):
        latent_distances = torch.norm(latent[:, None]-latent, dim=2, p=2)
        latent_distances = latent_distances/latent_norm
        sorted, indices = torch.sort(latent_distances)

        kNN_mask = torch.zeros((latent.size(0), latent.size(0),)).scatter(1, indices[:, 1:(self.k+1)], 1)
        return latent_distances, kNN_mask


    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return torch.square((signature1-signature2)).sum()


    @staticmethod
    def sig_error2(signature1, signature2):
        """Compute distance between two topological signatures. Only consider distance if sig1 > sig2"""
        return (torch.clamp((signature1 - signature2),min = 0)**2).sum(dim=-1)

    def _count_matching_pairs(self, mask_X, mask_Z):
        """
        Computes fraction of matched pairs
        :param mask_X:
        :param mask_Z:
        :return:
        """
        mask_diff = mask_Z-mask_X
        return ((mask_Z.size(0)*self.k)-(mask_diff != 0).sum()*0.5)/(mask_Z.size(0)*self.k)

    def forward(self, latent,latent_norm, dist_X, pair_mask_X):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        dist_Z, pair_mask_Z = self._get_pairings_dist_Z(latent,latent_norm)


        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(pair_mask_X,pair_mask_Z)
        }
        if self.match_edges == 'symmetric':
            # L_X->Z
            sig1 = dist_X.mul(pair_mask_X)
            sig1_2 = dist_Z.mul(pair_mask_X)

            distance1_2 = torch.square((sig1-sig1_2)).sum()

            # L_Z->X
            sig2 = dist_Z.mul(pair_mask_Z)
            sig2_1 = dist_X.mul(pair_mask_Z)

            distance2_1 = torch.square((sig2_1-sig2)).sum()

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1
        elif self.match_edges == 'push1':
            # L_X->Z: same as for 'symmetric'
            # L_Z->X: pushes pairs apart that are closer together in the Z than in X,
            # but does NOT pull pairs together that are closer in X than in Z

            # L_X->Z
            sig1 = dist_X.mul(pair_mask_X)
            sig1_2 = dist_Z.mul(pair_mask_X)

            distance1_2 = torch.square((sig1-sig1_2)).sum()

            # L_Z->X
            sig2 = dist_Z.mul(pair_mask_Z)
            sig2_1 = dist_X.mul(pair_mask_Z)

            distance2_1 = torch.square(torch.clamp((sig2_1-sig2),min = 0)).sum()

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        else:
            raise ValueError

        return distance, distance_components
