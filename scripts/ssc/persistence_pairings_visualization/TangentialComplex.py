import gudhi

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll

if __name__ == "__main__":
    dataset_sampler = SwissRoll()
    n_points = 2048
    seed = 13
    samples, color = dataset_sampler.sample(n_points, seed=seed)

    tc = gudhi.TangentialComplex(intrisic_dim=1,points = samples)
    tc.compute_tangential_complex()
    simplex_tree = tc.create_simplex_tree()

    print(simplex_tree.get_skeleton(1))

    skeleton_sorted = sorted(simplex_tree.get_skeleton(1), key = lambda t: t[1])

    pairings = []
    for element in skeleton_sorted:
        pair = element[0]
        if len(pair) == 2 and element[1] == 0:
            print(pair)

            pairings.append(pair)

    make_plot(samples, pairings, color, name='witness_TEST')