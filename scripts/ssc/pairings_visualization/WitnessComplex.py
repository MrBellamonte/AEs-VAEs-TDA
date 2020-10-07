import math

import gudhi

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

# dataset_sampler = SwissRoll()
# data, color = dataset_sampler.sample(100, seed = 10)
#
# dataset_sampler = SwissRoll()
# data, color = dataset_sampler.sample(1000)
#
# witnesses = data
# landmarks = gudhi.pick_n_random_points(points=witnesses, nb_points=32)
#
# witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
# simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = 10000,
#                                                    limit_dimension=2)
if __name__ == "__main__":
    dataset_sampler = SwissRoll()

    n_samples_array = [48] #[32,48,64,96,128]
    n_witnesses_array = [256] #[256, 512]
    seeds = [13]#[10,13,20]
    for n_witnesses in n_witnesses_array:
        for seed in seeds:
            for n_samples in n_samples_array:

                #name = 'witness_nl{}_nw{}_seed{}'.format(n_samples, n_witnesses, seed)
                name = 'TEST_witness_nl{}_nw{}_seed{}'.format(n_samples, n_witnesses, seed)

                landmarks, color = dataset_sampler.sample(n_samples, seed=seed)
                witnesses, _ = dataset_sampler.sample(n_witnesses, seed=(seed + 17))
                witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
                simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = 1000,
                                                                   limit_dimension=1)

                skeleton = simplex_tree.get_skeleton(1)
                pairss = []
                for element in skeleton:
                    if len(element[0]) == 2:
                        if element[1] == 0:
                            pairss.append(element[0])
                print('Skeleton W-Euclidian: {}'.format(skeleton))
                print('Pairs W-Euclidian: {}'.format(pairss))
                print('Number of pairs: {}'.format(len(pairss)))



                skeleton_sorted = sorted(skeleton, key = lambda t: t[1])

                indices = list(range(0,n_samples))
                #print(skeleton_sorted)
                pairings = []
                for element in skeleton_sorted:
                    pair = element[0]
                    if len(pair) == 2:
                        #print(pair)
                        add = False
                        if pair[0] in indices:
                            add = True
                            indices.remove(pair[0])
                        if pair[1] in indices:
                            add = True
                            indices.remove(pair[1])

                        if add:
                            pairings.append(pair)

                        if len(indices)==0:
                            break
                    else:
                        pass







                simplex_tree.persistence()
                print(simplex_tree.persistence_pairs())
                pairs_filtered = []
                indices = list(range(0, n_samples))
                for element in simplex_tree.get_filtration():
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


                make_plot(landmarks, pairs_filtered, color, name=name)