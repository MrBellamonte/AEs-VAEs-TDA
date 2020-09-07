import torch

from src.topology.witness_complex import WitnessComplex


def compute_wc_offline(dataset, data_loader, batch_size, method_args, name = ''):
    print('Compute Witness Complex Pairings {name}'.format(name = name))

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
        landmarks_dist = torch.tensor(witness_complex.landmarks_dist)
        sorted, indices = torch.sort(landmarks_dist)
        kNN_mask = torch.zeros(
            (batch_size, batch_size), device='cpu'
        ).scatter(1, indices[:, 1:(method_args['k']+1)], 1)
        dist_X_all[batch, :, :] = landmarks_dist
        pair_mask_X_all[batch, :, :] = kNN_mask

    return dist_X_all, pair_mask_X_all