"""Topolologically regularized autoencoder using approximation."""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        self.normalize = toposig_kwargs['normalize']
        self.topo_sig = TopologicalSignatureDistanceWC(**toposig_kwargs)
        self.autoencoder = autoencoder

        if self.normalize:
            self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)
        else:
            self.latent_norm = 1

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x, dist_X, pair_mask_X,norm_X, labels = None):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """

        dist_X = torch.norm(x[:, None]-x, dim=2, p=2)

        if self.normalize:
            dist_X = dist_X / norm_X
        else:
            pass

        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)
        latent = self.autoencoder.encode(x)
        topo_error, topo_error_components = self.topo_sig(latent,self.latent_norm, dist_X, pair_mask_X, labels = labels)


        # normalize topo_error according to batch_size
        batch_size = x.size(0)
        topo_error = topo_error / (float(batch_size)*self.k)
        loss = self.lam_r * ae_loss + self.lam_t * topo_error
        loss_components = {
            'loss.autoencoder': ae_loss,
            'loss.topo_error': topo_error
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

    def __init__(self, k, match_edges, mu_push,normalize = True):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.k = k
        self.match_edges = match_edges
        self.mu_push = mu_push



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
        mask_X1 = mask_X.bool()
        mask_X2 = mask_X.t().bool()
        mask_Xtot = mask_X1 + mask_X2
        mask_Xtot = mask_Xtot.float()
        mask_Z1 = mask_Z.bool()
        mask_Z2 = mask_Z.t().bool()
        mask_Ztot = mask_Z1+mask_Z2
        mask_Ztot = mask_Ztot.float()

        tot_pairings = mask_Xtot.sum()
        missed_pairings = ((mask_Xtot-mask_Ztot) == 1).sum()


        return (tot_pairings-missed_pairings)/tot_pairings

    def _get_count_nonmatching_pairs(self, mask_X, mask_Z):
        mask_Xtot = (mask_X.bool()+mask_X.bool().t()).int()
        mask_Ztot = (mask_Z.bool()+mask_Z.bool().t()).int()

        mask_Z_nonmatching = ((mask_Ztot - mask_Xtot)==1).int()

        count = int(mask_Z_nonmatching.sum())/int(mask_Ztot.sum())

        return count, mask_Z_nonmatching

    def forward(self, latent,latent_norm, dist_X, pair_mask_X, labels = None):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        dist_Z, pair_mask_Z = self._get_pairings_dist_Z(latent,latent_norm)

        non_matching_pairs, mask_Z_nonmatching = self._get_count_nonmatching_pairs(pair_mask_X, pair_mask_Z)
        distance_components = {
            'metrics.notmatched_pairs_0D': non_matching_pairs
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


            # # CHECK GRAD CALCULATION
            # res12 = torch.autograd.gradcheck(torch.square, (sig2-sig2_1),
            #                                raise_exception=True,eps=1e-4, atol=1e-4)
            # res21 = torch.autograd.gradcheck(torch.square, (sig2_1-sig2),
            #                                raise_exception=True,eps=1e-4, atol=1e-4)
            # print('Gradient 1-2:{}'.format(res12))
            # print('Gradient 2-1:{}'.format(res21))

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1
        elif self.match_edges == 'push':
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
        elif self.match_edges == 'push_active':
            # L_X->Z: same as for 'symmetric'
            # L_Z->X: pushes pairs apart that are closer together in the Z than in X,
            # but does NOT pull pairs together that are closer in X than in Z

            # L_X->Z
            sig1 = dist_X.mul(pair_mask_X)
            sig1_2 = dist_Z.mul(pair_mask_X)

            distance1_2 = torch.square((sig1-sig1_2)).sum()

            # L_Z->X
            dist_X_ref = dist_X.detach().clone()
            dist_X_ref = self.mu_push*dist_X_ref

            sig2 = dist_Z.mul(mask_Z_nonmatching)
            sig2_ref = dist_X_ref.mul(mask_Z_nonmatching)

            distance2_1 = torch.square(torch.clamp((sig2_ref-sig2),min = 0)).sum()
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1


            distance = distance1_2 + distance2_1

        elif self.match_edges == 'push2':
            # L_X->Z: same as for 'symmetric'
            # L_Z->X: pushes pairs apart that are closer together in the Z than in X,
            # but does NOT pull pairs together that are closer in X than in Z
            #todo: push apart actively the pairs that appear in Z but not in X

            # L_X->Z
            sig1 = dist_X.mul(pair_mask_X)
            sig1_2 = dist_Z.mul(pair_mask_X)

            distance1_2 = torch.square((sig1-sig1_2)).sum()

            # L_Z->X
            pair_mask_X2 = torch.ones_like(pair_mask_X) - pair_mask_X
            sig_push = dist_Z.mul(pair_mask_X2)
            sig_push_sum = torch.square(sig_push).sum()

            # L_Z->X
            sig2 = dist_Z.mul(pair_mask_Z)
            sig2_1 = dist_X.mul(pair_mask_Z)

            distance2_1 = 100000/sig_push_sum + torch.square(torch.clamp((sig2_1-sig2),min = 0)).sum()

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1
        elif 'verification':
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

            # reformat pairs
            ind_Z = np.where(pair_mask_Z == 1)
            ind_Z = np.column_stack((ind_Z[0], ind_Z[1]))

            ind_X = np.where(pair_mask_X == 1)
            ind_X = np.column_stack((ind_X[0], ind_X[1]))

            lanten_np = latent.detach().numpy()
            if labels is None:
                pass
            else:
                data = pd.DataFrame({'x': lanten_np[:, 0], 'y': lanten_np[:, 1], 'label': labels})

                for pair in ind_Z:
                    plt.plot(lanten_np[pair, 0], lanten_np[pair, 1], color='green',zorder = 4)
                for pair in ind_X:
                    plt.plot(lanten_np[pair, 0], lanten_np[pair, 1], color='blue',zorder = 3)
                sns.scatterplot('x', 'y', hue='label', data=data, palette=sns.color_palette('Spectral', len(np.unique(labels))), zorder = 10, legend = None)
                sns.despine(left=True, bottom=True)
                plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False,
                                left=False)

                plt.show()
                plt.close()

        else:
            raise ValueError

        return distance, distance_components
