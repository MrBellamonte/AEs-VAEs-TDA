import numpy as np
import pandas as pd
import torch

from src.topology.witness_complex import WitnessComplex
from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot


def compute_wc_offline(dataset, data_loader, batch_size, method_args, name='', verfication = False):
    print('Compute Witness Complex Pairings {name}'.format(name=name))

    dist_X_all = torch.ones((len(data_loader), batch_size, batch_size))
    pair_mask_X_all = torch.ones((len(data_loader), batch_size, batch_size))

    for batch, (img, label) in enumerate(data_loader):
        witness_complex = WitnessComplex(img, dataset[:][:][0])

        if method_args['n_jobs'] > 1:
            witness_complex.compute_simplicial_complex_parallel(d_max=1,
                                                                r_max=method_args['r_max'],
                                                                create_simplex_tree=False,
                                                                create_metric=True,
                                                                n_jobs=method_args['n_jobs'])
        else:
            witness_complex.compute_simplicial_complex(d_max=1,
                                                       r_max=method_args['r_max'],
                                                       create_simplex_tree=False,
                                                       create_metric=True)

        if witness_complex.check_distance_matrix:
            pass
        else:
            print('WARNING: choose higher r_max')
        landmarks_dist = torch.tensor(witness_complex.landmarks_dist)
        sorted, indices = torch.sort(landmarks_dist)
        kNN_mask = torch.zeros((batch_size, batch_size), device='cpu').scatter(1, indices[:, 1:(method_args['k']+1)], 1)
        dist_X_all[batch, :, :] = landmarks_dist
        pair_mask_X_all[batch, :, :] = kNN_mask

        if method_args['match_edges'] == 'verification' and verfication:
            ind_X = np.where(pair_mask_X_all[batch, :, :] == 1)
            ind_X = np.column_stack((ind_X[0], ind_X[1]))

            make_plot(img, ind_X, label,'name', path_root = None, knn = False)




    return dist_X_all, pair_mask_X_all


def compute_kNN_mask(latent, latent_norm, k):
    latent_distances = torch.norm(latent[:, None]-latent, dim=2, p=2)
    
    latent_distances = latent_distances/latent_norm
    sorted, indices = torch.sort(latent_distances)
    kNN_mask = torch.zeros((latent.size(0), latent.size(0))).scatter(1, indices[:, 1:(k+1)],1)
    return latent_distances, kNN_mask
