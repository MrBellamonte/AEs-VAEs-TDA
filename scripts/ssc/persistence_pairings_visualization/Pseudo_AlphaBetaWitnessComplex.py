import gudhi
from scipy.spatial.distance import cdist

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

NLT_TEST = [
    [[0,1],[3,9],[1,29],[2,34]],
    [[1,1],[0,16],[2,26],[3,39]],
    [[2,4],[1,9],[3,29],[0,34]],
    [[3,4],[2,9],[0,29],[1,34]]
]


def get_nlt(landmarks, witnesses):
    nlt = []

    for i in range(len(witnesses)):
        witness = witnesses[i,:].reshape((3, 1)).transpose()
        distances = cdist(landmarks, witness)
        temp = []
        for count, value in enumerate(distances):
            temp.append([count, float(value)])

        nlt.append(temp)

    return nlt





def filter_alpha(nlt, alpha):
    nlt_filtered = []
    for witness_list in nlt:
        temp_witness = []
        for pair in witness_list:
            if pair[1]<=alpha:
                temp_witness.append(pair)
        nlt_filtered.append(temp_witness)
    
    return nlt_filtered

def get_pairings(skeleton, indices_remaining, pairings):
    skeleton_sorted = sorted(skeleton, key=lambda t: t[1])

    for element in skeleton_sorted:
        pair = element[0]
        if len(pair) == 2:
            add = False
            if pair[0] in indices_remaining:
                add = True
                indices_remaining.remove(pair[0])
            if pair[1] in indices_remaining:
                add = True
                indices_remaining.remove(pair[1])

            if add:
                pairings.append(pair)

            if len(indices_remaining) == 0:
                break
        else:
            pass

    return indices_remaining, pairings



def pseudo_alpha_beta_witness_complex(n_lm, alpha_beta, n_landmarks):

    indices_remaining = list(range(0,n_landmarks))
    pairings = []

    keep_searching_pairs = True


    i = 0
    while keep_searching_pairs:

        alpha, beta = alpha_beta[i][0], alpha_beta[i][1]

        n_lm_current = filter_alpha(n_lm, alpha)

        witness_complex = gudhi.WitnessComplex(nearest_landmark_table = n_lm_current)
        simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=beta,limit_dimension=1)
        skeleton = simplex_tree.get_skeleton(1)

        indices_remaining, pairings = get_pairings(skeleton, indices_remaining, pairings)

        i += 1

        if len(indices_remaining) == 0:
            keep_searching_pairs = False
        elif len(alpha_beta) == i:
            keep_searching_pairs = False
            print('Stopped early')
        else:
            pass

    return pairings


def count_pairings(n_landmarks, pairings):
    pair_conc = []
    for pair in pairings:
        pair_conc.append(pair[0])
        pair_conc.append(pair[1])

    pair_conc = set(pair_conc)

    print('Should: {} ---- Is: {}'.format(n_landmarks,len(pair_conc)))

if __name__ == "__main__":
    alpha_beta = []
    for i in range(20):
        alpha_beta.append([2**(0.25*i+0.0000001),2**(0.25*i)])

    n_samples_array = [32,48,64,96,128]
    n_witnesses_array = [256, 512]
    seeds = [10,13,20]
    for n_witnesses in n_witnesses_array:
        for seed in seeds:
            for n_samples in n_samples_array:

                name = 'witness_alphabeta_nl{}_nw{}_seed{}'.format(n_samples, n_witnesses, seed)
                dataset_sampler = SwissRoll()
                n_landmarks = n_samples
                seed = seed
                landmarks, color = dataset_sampler.sample(n_landmarks, seed = seed)
                witnesses, _ = dataset_sampler.sample(n_witnesses, seed=(seed+17))


                nlt = get_nlt(landmarks, witnesses)


                pairings = pseudo_alpha_beta_witness_complex(nlt, alpha_beta, n_landmarks)

                count_pairings(n_landmarks, pairings)

                make_plot(landmarks, pairings, color, name=name)



