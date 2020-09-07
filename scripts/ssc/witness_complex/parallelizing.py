import math
import os
import multiprocessing as mp
import time

import numpy as np

def compute_simplicial_complex_parallel(self, d_max, create_metric = False, r_max = None, create_simplex_tree = False, n_jobs = 1):
    distances_all = np.random.randint(size=(12, 3), low=0, high=3)
    pool = mp.Pool(processes=n_jobs)
    distances_chunk = np.array_split(distances_all, n_jobs)

    def process_wc(distances, r_max = 1000, d_max = math.inf):
        landmarks_dist = np.ones((distances.shape[1], distances.shape[1]))*1000000
        simplicial_complex = []

        def update_register_simplex(register, i_add, i_dist, max_dim):
            register_add = []
            simplex_add = []
            for element in register:
                if len(element) < max_dim+1:
                    element_copy = element.copy()
                    element_copy.append(i_add)
                    register_add.append(element_copy)
                    simplex_add.append([element_copy, i_dist])
            return register_add, simplex_add

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
            register = []
            for i in range(len(sorted_row)):
                register_add, simplex_add = update_register_simplex(register.copy(),
                                                                          sorted_row[i][0],
                                                                          sorted_row[i][1], d_max)

                landmarks_dist = update_landmark_dist(landmarks_dist, simplex_add)
                register += register_add
                register.append([sorted_row[i][0]])
                simplices_temp += simplex_add

            simplicial_complex += simplices_temp

        return landmarks_dist, simplicial_complex


    multi_result = [pool.apply_async(process_wc, (distance_matrix, 10, math.inf))
                        for distance_matrix in distances_chunk]

    results = [p.get() for p in multi_result]

    pool.close()


    simplicial_complex, landmarks_dist = combine_results(results)




if __name__ == '__main__':

    distances_all = np.random.randint(size = (1024,256), low = 0, high = 3)

    start = time.time()
    for row_i in range(distances_all.shape[0]):
        row = distances_all[row_i, :]

        # sort row by landmarks witnessed #todo implement as numpy map?
        sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])

    end = time.time()
    print(end-start)


    # qout = mp.Queue()
    # processes = [mp.Process(target=process_wc, args=(distance_matrix, qout))
    #              for distance_matrix in distances_chunk]
    #
    # for p in processes:
    #     p.start()
    #
    # for p in processes:
    #     p.join()

    # result = [qout.get() for p in processes]
    # print(result)

#     pool = mp.Pool(processes=n_jobs)
#     pool.close()
#     pool = mp.Pool(processes=n_jobs)
#     distances_chunk = np.array_split(distances_all, n_jobs)
#
#
#     multi_result = [pool.apply_async(process_wc, (distance_matrix, 10, math.inf))
#                  for distance_matrix in distances_chunk]
# #    result = [x for p in multi_result for x in p.get()]
#
#     results = [p.get() for p in multi_result]
#
#     pool.close()
#     var = 1
#
#     simplicial_complex, landmarks_dist = combine_results(results)







    # pool = Pool(os.cpu_count())                       # Create a multiprocessing Pool
    # pool.map(process_image, data_inputs)