import math

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

from scripts.ssc.persistence_pairings_visualization.Pseudo_AlphaBetaWitnessComplex import \
    count_pairings
from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll


def wl_table(witnesses, landmarks):
    return pairwise_distances(witnesses,landmarks)


def update_register_simplex(register, i_add, i_dist, max_dim = math.inf):
    register_add = []
    simplex_add = []
    for element in register:
        if len(element)< max_dim:
            element_copy = element.copy()
            element_copy.append(i_add)
            register_add.append(element_copy)
            simplex_add.append([element_copy, i_dist])
    return register_add, simplex_add


def get_pairs_0(distances):
    simplices = []
    for row_i in range(distances.shape[0]):
        col = distances[row_i,:]
        sort_col = sorted([*enumerate(col)], key=lambda x: x[1])


        simplices_temp = []
        register = []
        for i in range(len(sort_col)):
            register_add, simplex_add = update_register_simplex(register.copy(), sort_col[i][0],sort_col[i][1],2)

            register += register_add
            register.append([sort_col[i][0]])
            simplices_temp += simplex_add

        simplices += simplices_temp



    return sorted(simplices, key=lambda t: t[1])



def get_pairs_1(distances, landmarks):
    pairs = []
    for row_i in range(distances.shape[0]):
        temp = []
        col = distances[row_i,:]
        sort_col = sorted([*enumerate(col)], key=lambda x: x[1])
        i1, i2 = sort_col[0][0],sort_col[1][0]
        dist1, dist2 = sort_col[0][1], sort_col[1][1]
        temp.append([i1,i2])
        temp.append((dist1+dist2))

        pairs.append(temp)

    return sorted(pairs, key=lambda t: t[1])

def get_persistence_pairs(pairs, n_landmarks):
    indices = list(range(0, n_landmarks))
    pairs_filtered = []
    for element in pairs:
        pair = element[0]
        if len(pair) == 2:
            # print(pair)
            add = False
            if pair[0] in indices:
                add = True
                indices.remove(pair[0])
            if pair[1] in indices:
                add = True
                indices.remove(pair[1])

            if add:
                pairs_filtered.append(pair)

            if len(indices) == 0:
                break
        else:
            pass

    return pairs_filtered


if __name__ == "__main__":
    # n_samples_array = [32,48,64,96,128]
    # n_witnesses_array = [2048,4096]
    # seeds = [10,13,20]
    # for n_witnesses in n_witnesses_array:
    #     for seed in seeds:
    #         for n_samples in n_samples_array:
    #
    #             name = 'witness_ssc_nl{}_nw{}_seed{}'.format(n_samples, n_witnesses, seed)
    #             dataset_sampler = SwissRoll()
    #             n_landmarks = n_samples
    #             seed = seed
    #             landmarks, color = dataset_sampler.sample(n_landmarks, seed = seed)
    #             witnesses, _ = dataset_sampler.sample(n_witnesses, seed=(seed+17))
    #
    #
    #             distances = wl_table(witnesses,landmarks)
    #             pairs = get_pairs_1(distances, landmarks)
    #
    #             pairs_filtered = get_persistence_pairs(pairs, n_samples)
    #
    #             count_pairings(n_samples, pairs_filtered)
    #             make_plot(landmarks, pairs_filtered, color, name=name)

    n_samples_array = [32,48,64,96,128]
    n_witnesses_array = [256,512,1024]
    seeds = [10,13,20]
    n_samples_array = [64]
    n_witnesses_array = [512]
    seeds = [27]
    for n_witnesses in n_witnesses_array:
        for seed in seeds:
            for n_samples in n_samples_array:

                name = 'witness_ssc_corrected_nl{}_nw{}_seed{}'.format(n_samples, n_witnesses, seed)
                dataset_sampler = SwissRoll()
                n_landmarks = n_samples
                seed = seed
                landmarks, color = dataset_sampler.sample(n_landmarks, seed = seed)
                witnesses, _ = dataset_sampler.sample(n_witnesses, seed=(seed+17))


                distances = wl_table(witnesses,landmarks)
                pairs = get_pairs_0(distances)

                pairs_filtered = get_persistence_pairs(pairs, n_samples)

                count_pairings(n_samples, pairs_filtered)
                make_plot(landmarks, pairs_filtered, color, name=name)

