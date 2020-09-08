import math

import dill
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
#dirty fix since gudhi cannot be installed on euler...
try:
    import gudhi
except:
    print('Failed to import gudhi')
from sklearn.metrics import pairwise_distances

#hard-coded
MAX_DIST_INIT = 1000000


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

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)
def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


class WitnessComplex():
    __slots__=[
        'landmarks',
        'witnesses',
        'distances',
        'simplicial_complex',
        'landmarks_dist',
        'simplex_tree',
        'simplex_true_computed',
        'new',
        'metric_computed'
    ]

    def __init__(self, landmarks, witnesses,new=False):
        self.landmarks = landmarks
        self.witnesses = witnesses
        self.new = new
        self.metric_computed = False

        self.distances = pairwise_distances(witnesses,landmarks)

    def _update_register_simplex(self, simplicial_complex_temp, i_add, i_dist, max_dim=math.inf):

        simplex_add = []
        for e in simplicial_complex_temp:
            element = e[0]
            if len(element) < max_dim+1:
                element_copy = element.copy()
                element_copy.append(i_add)
                simplex_add.append([element_copy, i_dist])


        return simplex_add


    def _update_landmark_dist(self, landmarks_dist, simplex_add):

        for simplex in simplex_add:
            if len(simplex[0]) == 2:
                if landmarks_dist[simplex[0][0]][simplex[0][1]] > simplex[1]:
                    landmarks_dist[simplex[0][0]][simplex[0][1]] = simplex[1]
                    landmarks_dist[simplex[0][1]][simplex[0][0]] = simplex[1]
        return landmarks_dist

    def compute_simplicial_complex(self, d_max, create_metric = False, r_max = None, create_simplex_tree = False):
        if create_simplex_tree:
            simplicial_complex = []
            try:
                simplex_tree = gudhi.SimplexTree()
            except:
                print('Cannot create simplex tree')

        if create_metric:
            landmarks_dist = np.ones((len(self.landmarks),len(self.landmarks)))*MAX_DIST_INIT

        for row_i in range(self.distances.shape[0]):
            row = self.distances[row_i, :]

            # sort row by landmarks witnessed
            sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
            if r_max != None:
                sorted_row_new_temp = []
                for element in sorted_row:
                    if element[1] < r_max:
                        sorted_row_new_temp.append(element)
                sorted_row = sorted_row_new_temp

            simplices_temp = []
            for i in range(len(sorted_row)):
                simplex_add = self._update_register_simplex(simplices_temp.copy(), sorted_row[i][0],
                                                                        sorted_row[i][1], d_max)
                if create_metric:
                    landmarks_dist = self._update_landmark_dist(landmarks_dist,simplex_add)
                simplices_temp += simplex_add
                simplices_temp.append([[sorted_row[i][0]],sorted_row[i][1]])

        if create_simplex_tree:
            simplicial_complex += simplices_temp
            self.simplicial_complex = simplicial_complex
        if create_metric:
            np.fill_diagonal(landmarks_dist, 0)
            self.landmarks_dist = landmarks_dist
            self.metric_computed = True


        if create_simplex_tree:
            sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

            for i in range(len(self.landmarks)):
                simplex_tree.insert([i], filtration=0)

            for simplex in sorted_simplicial_compex:
                simplex_tree.insert(simplex[0], filtration = simplex[1])
                self.simplex_tree = simplex_tree
            self.simplex_true_computed = True



    def compute_simplicial_complex_parallel(self, d_max = math.inf, r_max=math.inf,
                                            create_simplex_tree=False,create_metric = False, n_jobs=-1):
        global process_wc

        def process_wc(distances, r_max=r_max, d_max=d_max, create_metric = create_metric, create_simplex_tree = create_simplex_tree):

            landmarks_dist = np.ones((distances.shape[1], distances.shape[1]))*MAX_DIST_INIT
            simplicial_complex = []

            def update_register_simplex(simplicial_complex, i_add, i_dist, max_dim):

                simplex_add = []
                for e in simplicial_complex:
                    element = e[0]
                    if len(element) < max_dim+1:
                        element_copy = element.copy()
                        element_copy.append(i_add)
                        simplex_add.append([element_copy, i_dist])
                return simplex_add

            def update_landmark_dist(landmarks_dist, simplex_add):
                for simplex in simplex_add:
                    if len(simplex[0]) == 2:
                        if landmarks_dist[simplex[0][0]][simplex[0][1]] > simplex[1]:
                            landmarks_dist[simplex[0][0]][simplex[0][1]] = simplex[1]
                            landmarks_dist[simplex[0][1]][simplex[0][0]] = simplex[1]
                return landmarks_dist

            for row_i in range(distances.shape[0]):
                row = distances[row_i, :]

                # sort row by landmarks witnessed #todo implement as numpy map?
                sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
                if r_max != None:
                    sorted_row_new_temp = []
                    for element in sorted_row:
                        if element[1] < r_max:
                            sorted_row_new_temp.append(element)
                    sorted_row = sorted_row_new_temp

                simplices_temp = []
                for i in range(len(sorted_row)):
                    simplex_add = update_register_simplex(simplices_temp.copy(),
                                                                        sorted_row[i][0],
                                                                        sorted_row[i][1], d_max)
                    if create_metric:
                        landmarks_dist = update_landmark_dist(landmarks_dist, simplex_add)
                    simplices_temp += simplex_add
                    simplices_temp.append([[sorted_row[i][0]],sorted_row[i][1]])

                if create_simplex_tree:
                    simplicial_complex += simplices_temp

            return landmarks_dist, simplicial_complex

        def combine_results(results, create_metric, create_simplex_tree):

            simplicial_complex = []

            for i, result in enumerate(results):
                if create_metric:
                    if i is 0:
                        landmarks_dist = result[0]
                    else:
                        landmarks_dist = np.dstack((landmarks_dist, result[0]))
                else:
                    landmarks_dist = result[0]

                if create_simplex_tree:
                    simplicial_complex.append(result[1])

            if create_metric:
                landmarks_dist = np.amin(landmarks_dist, axis=2)

            return simplicial_complex, landmarks_dist

        if create_simplex_tree:
            try:
                simplex_tree = gudhi.SimplexTree()
            except:
                print('Cannot create simplex tree')

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        pool = mp.Pool(processes=n_jobs)
        distances_chunk = np.array_split(self.distances, n_jobs)


        results = pool.map(process_wc, distances_chunk)

        pool.close()

        simplicial_complex, landmarks_dist = combine_results(results, create_metric, create_simplex_tree)

        if create_simplex_tree:
            sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

            for i in range(len(self.landmarks)):
                simplex_tree.insert([i], filtration=0)

            for simplex in sorted_simplicial_compex:
                simplex_tree.insert(simplex[0], filtration = simplex[1])
                self.simplex_tree = simplex_tree
            self.simplex_true_computed = True



        self.simplicial_complex = simplicial_complex
        np.fill_diagonal(landmarks_dist, 0)
        self.landmarks_dist = landmarks_dist
        self.metric_computed = True



    def get_diagram(self, show = False, path_to_save = None):
        assert self.simplex_true_computed
        fig, ax = plt.subplots()


        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, axes = ax)

        if show:
            plt.show()

        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=200)
        plt.close()


    def check_distance_matrix(self):
        assert self.metric_computed

        return not np.any(self.landmarks_dist == MAX_DIST_INIT)
