"""Topolologically regularized autoencoder using approximation."""
import numpy as np
import torch
import torch.nn as nn


from .topology import PersistentHomologyCalculation
from src.models.autoencoder.base import AutoencoderModel


class TopologicallyRegularizedAutoencoder(AutoencoderModel):
    """Topologically regularized autoencoder."""

    def __init__(self,autoencoder, lam_t=1.,lam_r=1., toposig_kwargs=None):
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
        self.push_edges = (toposig_kwargs['match_edges'] == 'asymmetric_push')
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.autoencoder = autoencoder
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x,x_distances = None, mu = 0):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """
        latent = self.autoencoder.encode(x)

        if x_distances is None:
            x_distances = self._compute_distance_matrix(x)
        else:
            pass


        dimensions = x.size()
        if len(dimensions) == 4:
            # If we have an image dataset, normalize using theoretical maximum
            batch_size, ch, b, w = dimensions
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2**2 * ch * b * w) ** 0.5
            x_distances = x_distances / max_distance
        else:
            # Else just take the max distance we got in the batch
            x_distances = x_distances / x_distances.max()

        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)

        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances, mu)


        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
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


class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def sig_error2(signature1, signature2):
        """Compute distance between two topological signatures. Only consider distance if sig1 > sig2"""
        return (torch.clamp((signature1 - signature2),min = 0)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(float(len(to_set(pairs1).intersection(to_set(pairs2))))/float(len(to_set(pairs1))))

    @staticmethod
    def _get_pairdiff(pairs_base, pairs_sub):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        l1 = to_set(pairs_base[0])
        l2 = to_set(pairs_sub[0])
        ldiff = l1 - l2
        if ldiff == set():
            return None
        else:
            lldiff_np = np.array(list(ldiff))
            return [lldiff_np,0]


    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2, mu):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'asymmetric':
            # Only preserve distances from X->Z
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)


            distance1_2 = self.sig_error(sig1, sig1_2)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = 0

            distance = distance1_2

        elif self.match_edges == 'asymmetric2':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error2(sig2_1,sig2)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2+distance2_1

        elif self.match_edges == 'asymmetric_push':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            pairs1_0, pairs1_1 = pairs1
            #idx = (pairs1_0[:, 0], pairs1_0[:, 1])
            distance_push = torch.clamp(distances2,min=0.1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance_push.pow_(-1).sum()

            distance = distance1_2 + mu*distance_push.pow_(-1).sum()
        elif self.match_edges == 'asymmetric_push2':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            pairs_push = self._get_pairdiff(pairs2,pairs1)
            if pairs_push == None:
                loss_push = 0
            else:
                sig2 = self._select_distances_from_pairs(distances2, pairs_push)
                distance_push = torch.clamp(sig2, min=0.5)
                loss_push = distance_push.pow_(-1).sum()
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)

            distance1_2 = self.sig_error(sig1, sig1_2)



            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = loss_push

            distance = distance1_2 + loss_push
        elif self.match_edges == 'asymmetric_push3':

            B = 2

            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)

            distance1_2 = self.sig_error(sig1, sig1_2)

            distances2_selected = distances2
            distances2_selected[(pairs1[0][:, 0], pairs1[0][:, 1])] = 0
            distances2_selected = distances2_selected - B
            distances2_selected = torch.clamp(distances2_selected, max = 0)
            distances2_selected = distances2_selected+B

            loss_push = distances2_selected.sum()


            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = mu*loss_push

            distance = distance1_2 - mu*loss_push
        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1
        elif self.match_edges == 'no':
            distance_components['metrics.distance1-2'] = 0
            distance_components['metrics.distance2-1'] = 0

            distance = 0

        return distance, distance_components
