from sklearn.neighbors import NearestNeighbors

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll
from src.topology.witness_complex import WitnessComplex

PATH = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/witness_complex_k/'

if __name__ == "__main__":

    n_landmarks = 512
    n_witnesses = 2048
    seed = 0

    dataset_sampler = SwissRoll()
    landmarks, color = dataset_sampler.sample(n_landmarks, seed=seed)
    witnesses, _ = dataset_sampler.sample(n_witnesses, seed=(seed+17))

    witness_complex = WitnessComplex(landmarks,witnesses)
    witness_complex.compute_simplicial_complex(1,True,r_max=7)



    for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        name = 'nl{}_nw{}_k{}_seed{}'.format(n_landmarks,n_witnesses,k,seed)

        neigh = NearestNeighbors(n_neighbors=(k+1), metric='precomputed').fit(witness_complex.landmarks_dist)
        distances, pairings = neigh.kneighbors(witness_complex.landmarks_dist)
        print(distances)
        print(pairings)

        make_plot(landmarks, pairings, color,name, path_root = PATH, knn = True)



