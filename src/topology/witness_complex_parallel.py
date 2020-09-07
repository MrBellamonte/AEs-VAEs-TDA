import math

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from sklearn.metrics import pairwise_distances

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

class WitnessComplexPara():
    __slots__=[
        'landmarks',
        'witnesses',
        'distances',
        'simplicial_complex',
        'landmarks_dist',
        'simplex_tree',
        'simplex_true_computed',
        'n_jobs'
    ]

    def __init__(self, landmarks, witnesses, n_jobs = 2):
        self.landmarks = landmarks
        self.witnesses = witnesses
        self.n_jobs = n_jobs
        self.distances = pairwise_distances(witnesses,landmarks)

    def _update_register_simplex(self,register, i_add, i_dist, max_dim=math.inf):
        register_add = []
        simplex_add = []
        for element in register:
            if len(element) < max_dim+1:
                element_copy = element.copy()
                element_copy.append(i_add)
                register_add.append(element_copy)
                simplex_add.append([element_copy, i_dist])
        return register_add, simplex_add

    def _update_landmark_dist(self, landmarks_dist, simplex_add):
        for simplex in simplex_add:
            if len(simplex[0]) == 2:
                if landmarks_dist[simplex[0][0]][simplex[0][1]] > simplex[1]:
                    landmarks_dist[simplex[0][0]][simplex[0][1]] = simplex[1]
                    landmarks_dist[simplex[0][1]][simplex[0][0]] = simplex[1]
        return landmarks_dist


    def apply_along_axis(self, row,r_max,d_max, threads):
        import numpy as np
        import threading

        def threaded_process(items_chunk):
            """ Your main process which runs in thread for each chunk"""
            for item in items_chunk:
                try:
                    api.my_operation(item)
                except Exception:
                    print('error with item')

        n_threads = 20
        # Splitting the items into chunks equal to number of threads
        array_chunk = np.array_split(input_image_list, n_threads)
        thread_list = []
        for thr in range(n_threads):
            thread = threading.Thread(target=threaded_process, args=(array_chunk[thr]), )
            thread_list.append(thread)
            thread_list[thr].start()

        for thread in thread_list:
            thread.join()

    def compute_simplicial_complex(self, d_max, create_metric = False, r_max = None, create_simplex_tree = False):
        simplicial_complex = []

        if create_simplex_tree:
            simplex_tree = gudhi.SimplexTree()

        if create_metric:
            landmarks_dist = np.ones((len(self.landmarks),len(self.landmarks)))*1000000

        for row_i in range(self.distances.shape[0]):
            #todo parallelize
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
            register = []
            for i in range(len(sorted_row)):
                register_add, simplex_add = self._update_register_simplex(register.copy(), sorted_row[i][0],
                                                                    sorted_row[i][1], d_max)
                if create_metric:
                    landmarks_dist = self._update_landmark_dist(landmarks_dist,simplex_add)
                register += register_add
                register.append([sorted_row[i][0]])
                simplices_temp += simplex_add


            simplicial_complex += simplices_temp

        self.simplicial_complex = simplicial_complex
        if create_metric:
            np.fill_diagonal(landmarks_dist, 0)
            self.landmarks_dist = landmarks_dist


        if create_simplex_tree:
            sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

            for i in range(len(self.landmarks)):
                simplex_tree.insert([i], filtration=0)

            for simplex in sorted_simplicial_compex:
                simplex_tree.insert(simplex[0], filtration = simplex[1])
                self.simplex_tree = simplex_tree
            self.simplex_true_computed = True

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



