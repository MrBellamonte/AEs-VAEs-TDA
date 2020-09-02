import random

import gudhi
import numpy as np

from scripts.ssc.persistence_pairings_visualization.PEH_TopoAE_testing import make_plot
from src.datasets.datasets import SwissRoll

data = np.array([[0,0], [1,1],[1,2],[4,2]])

rips_complex = gudhi.RipsComplex(points = data)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
simplex_tree.persistence()
pers_pairs = simplex_tree.persistence_pairs()
result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)
print(pers_pairs)


landmarks = data
witnesses = np.array([[0,0], [1,1],[1,2],[4,2],[0.5,0.5], [1,1.5],[2.5,2]])

witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = 0,
                                                   limit_dimension=1)

skeleton = simplex_tree.get_skeleton(1)
pairs = []
for element in skeleton:
    if len(element[0]) == 2:
        if element[1] == 0:
            pairs.append(element[0])
print('Skeleton W-Euclidian: {}'.format(skeleton))
print('Pairs W-Euclidian: {}'.format(pairs))
simplex_tree.persistence(persistence_dim_max = True)
pers_pairs = simplex_tree.persistence_pairs()

print(pers_pairs)


result_str = 'Witness complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)

nlt = [
    [[0,1],[3,9],[1,29],[2,34]],
    [[1,1],[0,16],[2,26],[3,39]],
    [[2,4],[1,9],[3,29],[0,34]],
    [[3,4],[2,9],[0,29],[1,34]]
]

wcomp = gudhi.WitnessComplex(nlt)
simplex_tree = wcomp.create_simplex_tree(max_alpha_square=50)
skeleton = simplex_tree.get_skeleton(1)
pairs = []
for element in skeleton:
    if len(element[0]) == 2:
        if element[1] == 0:
            pairs.append(element[0])
print('Skeleton W-LIST: {}'.format(skeleton))
print('Pairs W-LIST: {}'.format(pairs))





dataset_sampler = SwissRoll()
n_landmarks = (64+64)
n_total = 4048*2*2*2*2
seed = 9
#landmarks_, color_ = dataset_sampler.sample(n_landmarks, seed = 9)
witnesses_, color_ = dataset_sampler.sample(n_total, seed = seed)

landmark_indices = random.sample(list(range(0,n_total)), n_landmarks)

#landmarks_ = gudhi.pick_n_random_points(points=witnesses_, nb_points=n_landmarks)

landmarks, color_landmarks = witnesses_[landmark_indices,:], color_[landmark_indices]

witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses_, landmarks=landmarks)
simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = 100,
                                                   limit_dimension=1)

skeleton = simplex_tree.get_skeleton(1)
pairs = []
for element in skeleton:
    if len(element[0]) == 2:
        if element[1] == 0:
            pairs.append(element[0])
print('Skeleton W-Euclidian: {}'.format(skeleton))
print('Pairs W-Euclidian: {}'.format(pairs))
print('Number of pairs: {}'.format(len(pairs)))

pair = []
filtration_value = []
for element in skeleton:
    filtration_value.append(element[1])
    pair.append(element[0])

skeleton_sorted = sorted(skeleton, key = lambda t: t[1])

indices = list(range(0,n_landmarks))

pairings = []
for element in skeleton_sorted:
    pair = element[0]
    if len(pair) == 2:
        print(pair)
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





make_plot(landmarks, pairings, color_landmarks, name = 'witness_TEST')




