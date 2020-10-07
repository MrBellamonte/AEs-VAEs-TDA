import numpy as np
import math
import gudhi
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from scripts.ssc.persistence_pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll


def DTM(X, query_pts, m):
    '''
    Compute the values of the DTM (with exponent p=2) of the empirical measure of a point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors

    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a kxd numpy array of query points
    m: parameter of the DTM in [0,1)

    Output:
    DTM_result: a kx1 numpy array contaning the DTM of the
    query points

    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTM_values = DTM(X, Q, 0.3)
    '''
    N_tot = X.shape[0]
    k = math.floor(m*N_tot)+1  # number of neighbors

    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)

    DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist, axis=1)/k)

    return (DTM_result)


def Filtration_value(p, fx, fy, d, n=10):
    '''
    Compute the filtrations values of the edge [x,y] in the weighted Rips filtration
    If p is not 1, 2 or 'np.inf, an implicit equation is solved
    The equation to solve is G(I) = d, where G(I) = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p)
    We use a dichotomic method

    Input:
    p: parameter of the weighted Rips filtration, in [1, +inf) or np.inf
    fx: filtration value of the point x
    fy: filtration value of the point y
    d: distance between the points x and y
    n: number of iterations of the dichotomic method

    Output:
    val : filtration value of the edge [x,y], i.e. solution of G(I) = d

    Example:
    Filtration_value(2.4, 2, 3, 5, 10)
    '''
    if p == np.inf:
        value = max([fx, fy, d/2])
    else:
        fmax = max([fx, fy])
        if d < (abs(fx**p-fy**p))**(1/p):
            value = fmax
        elif p == 1:
            value = (fx+fy+d)/2
        elif p == 2:
            value = np.sqrt(((fx+fy)**2+d**2)*((fx-fy)**2+d**2))/(2*d)
        else:
            Imin = fmax;
            Imax = (d**p+fmax**p)**(1/p)
            for i in range(n):
                I = (Imin+Imax)/2
                g = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p)
                if g < d:
                    Imin = I
                else:
                    Imax = I
            value = I
    return value


def WeightedRipsFiltration(X, F, p, dimension_max=2, filtration_max=np.inf):
    '''
    Compute the weighted Rips filtration of a point cloud, weighted with the
    values F, and with parameter p

    Input:
    X: a nxd numpy array representing n points in R^d
    F: an array of length n,  representing the values of a function on X
    p: a parameter in [0, +inf) or np.inf
    filtration_max: maximal filtration value of simplices when building the complex
    dimension_max: maximal dimension to expand the complex

    Output:
    st: a gudhi.SimplexTree
    '''
    N_tot = X.shape[0]
    distances = euclidean_distances(X)  # compute the pairwise distances
    st = gudhi.SimplexTree()  # create an empty simplex tree

    for i in range(N_tot):  # add vertices to the simplex tree
        value = F[i]
        if value < filtration_max:
            st.insert([i], filtration=F[i])
    for i in range(N_tot):  # add edges to the simplex tree
        for j in range(i):
            value = Filtration_value(p, F[i], F[j], distances[i][j])
            if value < filtration_max:
                st.insert([i, j], filtration=value)

    st.expansion(dimension_max)  # expand the simplex tree

    result_str = 'Weighted Rips Complex is of dimension '+repr(st.dimension())+' - '+ \
                 repr(st.num_simplices())+' simplices - '+ \
                 repr(st.num_vertices())+' vertices.'+ \
                 ' Filtration maximal value is '+str(filtration_max)+'.'
    print(result_str)

    return st


def DTMFiltration(X, m, p, dimension_max=2, filtration_max=np.inf):
    '''
    Compute the DTM-filtration of a point cloud, with parameters m and p

    Input:
    X: a nxd numpy array representing n points in R^d
    m: parameter of the DTM, in [0,1)
    p: parameter of the filtration, in [0, +inf) or np.inf
    filtration_max: maximal filtration value of simplices when building the complex
    dimension_max: maximal dimension to expand the complex

    Output:
    st: a gudhi.SimplexTree
    '''

    DTM_values = DTM(X, X, m)
    st = WeightedRipsFiltration(X, DTM_values, p, dimension_max, filtration_max)

    return st


def AlphaWeightedRipsFiltration(X, F, p, dimension_max=2, filtration_max=np.inf):
    '''
    /!\ this is a heuristic method, to speed-up the computation
    It computes the weighted Rips filtration as a subset of the alpha complex

    Input:
    X: a nxd numpy array representing n points in R^d
    F: an array of length n,  representing the values of a function on X
    p: a parameter in [0, +inf) or np.inf
    filtration_max: maximal filtration value of simplices when building the complex
    dimension_max: maximal dimension to expand the complex

    Output:
    st: a gudhi.SimplexTree
    '''
    N_tot = X.shape[0]
    distances = euclidean_distances(X)  # compute the pairwise distances

    st_alpha = gudhi.AlphaComplex(points=X).create_simplex_tree()
    st = gudhi.SimplexTree()  # create an empty simplex tree

    for simplex in st_alpha.get_skeleton(2):  # add vertices with corresponding filtration value
        if len(simplex[0]) == 1:
            i = simplex[0][0]
            st.insert([i], filtration=F[i])
        if len(simplex[0]) == 2:  # add edges with corresponding filtration value
            i = simplex[0][0]
            j = simplex[0][1]
            value = Filtration_value(p, F[i], F[j], distances[i][j])
            st.insert([i, j], filtration=value)

    st.expansion(dimension_max)  # expand the complex

    result_str = 'Alpha Weighted Rips Complex is of dimension '+repr(st.dimension())+' - '+ \
                 repr(st.num_simplices())+' simplices - '+ \
                 repr(st.num_vertices())+' vertices.'+ \
                 ' Filtration maximal value is '+str(filtration_max)+'.'
    print(result_str)

    return st


def AlphaDTMFiltration(X, m, p, dimension_max=2, filtration_max=np.inf):
    '''
    /!\ this is a heuristic method, to speed-up the computation
    It computes the DTM-filtration as a subset of the alpha complex

    Input:
    X: a nxd numpy array representing n points in R^d
    m: parameter of the DTM, in [0,1)
    p: parameter of the filtration, in [0, +inf) or np.inf
    filtration_max: maximal filtration value when building the complex
    dimension_max: maximal dimension to expand the complex

    Output:
    st: a gudhi.SimplexTree
    '''

    DTM_values = DTM(X, X, m)
    st = AlphaWeightedRipsFiltration(X, DTM_values, p, dimension_max, filtration_max)

    return st


def SampleOnCircle(N_obs=100, N_out=0, is_plot=False):
    '''
    Sample N_obs points (observations) points from the uniform distribution on the unit circle in R^2,
        and N_out points (outliers) from the uniform distribution on the unit square

    Input:
    N_obs: number of sample points on the circle
    N_noise: number of sample points on the square
    is_plot = True or False : draw a plot of the sampled points

    Output :
    data : a (N_obs + N_out)x2 matrix, the sampled points concatenated
    '''
    rand_uniform = np.random.rand(N_obs)*2-1
    X_obs = np.cos(2*np.pi*rand_uniform)
    Y_obs = np.sin(2*np.pi*rand_uniform)

    X_out = np.random.rand(N_out)*2-1
    Y_out = np.random.rand(N_out)*2-1

    X = np.concatenate((X_obs, X_out))
    Y = np.concatenate((Y_obs, Y_out))
    data = np.stack((X, Y)).transpose()

    if is_plot:
        fig, ax = plt.subplots()
        plt_obs = ax.scatter(X_obs, Y_obs, c='tab:cyan');
        plt_out = ax.scatter(X_out, Y_out, c='tab:orange');
        ax.axis('equal')
        ax.set_title(str(N_obs)+'-sampling of the unit circle with '+str(N_out)+' outliers')
        ax.legend((plt_obs, plt_out), ('data', 'outliers'), loc='lower left')
    return data


def SampleOnSphere(N_obs=100, N_out=0, is_plot=False):
    '''
    Sample N_obs points from the uniform distribution on the unit sphere in R^3,
        and N_out from the uniform distribution on the unit cube

    Input :
    N_obs: number of sample points on the sphere
    N_noise: number of sample points on the cube
    is_plot = True or False : draw a plot of the sampled points

    Output :
    data : a (N_obs + N_out)x3 matrix, the sampled points concatenated
    '''
    RAND_obs = np.random.rand(3, N_obs)*2-1
    norms = np.multiply(RAND_obs, RAND_obs)
    norms = np.sum(norms.T, 1).T
    X_obs = RAND_obs[0, :]/np.sqrt(norms)
    Y_obs = RAND_obs[1, :]/np.sqrt(norms)
    Z_obs = RAND_obs[2, :]/np.sqrt(norms)

    X_out = np.random.rand(N_out)*2-1
    Y_out = np.random.rand(N_out)*2-1
    Z_out = np.random.rand(N_out)*2-1

    X = np.concatenate((X_obs, X_out))
    Y = np.concatenate((Y_obs, Y_out))
    Z = np.concatenate((Z_obs, Z_out))
    data = np.stack((X, Y, Z)).transpose()

    if is_plot:
        fig = plt.figure();
        ax = fig.gca(projection='3d')
        plt_obs = ax.scatter(X_obs, Y_obs, Z_obs, c='tab:cyan')
        plt_out = ax.scatter(X_out, Y_out, Z_out, c='tab:orange')
        ax.set_title(str(N_obs)+'-sampling of the unit sphere with '+str(N_out)+' outliers')
        ax.legend((plt_obs, plt_out), ('data', 'outliers'), loc='lower left')
    return data


def SampleOnNecklace(N_obs=100, N_out=0, is_plot=False):
    '''
    Sample 4*N_obs points on a necklace in R^3,
        and N_out points from the uniform distribution on a cube

    Input :
    N_obs: number of sample points on the sphere
    N_noise: number of sample points on the cube
    is_plot = True or False : draw a plot of the sampled points

    Output :
    data : a (4*N_obs + N_out)x3 matrix, the sampled points concatenated
    '''

    X1 = SampleOnSphere(N_obs, N_out=0)+[2, 0, 0]
    X2 = SampleOnSphere(N_obs, N_out=0)+[-1, 2*.866, 0]
    X3 = SampleOnSphere(N_obs, N_out=0)+[-1, -2*.866, 0]
    X4 = 2*SampleOnCircle(N_obs, N_out=0)
    X4 = np.stack((X4[:, 0], X4[:, 1], np.zeros(N_obs))).transpose()

    data_obs = np.concatenate((X1, X2, X3, X4))

    X_out = 3*(np.random.rand(N_out)*2-1)
    Y_out = 3*(np.random.rand(N_out)*2-1)
    Z_out = 3*(np.random.rand(N_out)*2-1)
    data_out = np.stack((X_out, Y_out, Z_out)).transpose()

    data = np.concatenate((data_obs, data_out))

    if is_plot:
        fig = plt.figure();
        ax = fig.gca(projection='3d')
        plt_obs = ax.scatter(data_obs[:, 0], data_obs[:, 1], data_obs[:, 2], c='tab:cyan')
        plt_out = ax.scatter(X_out, Y_out, Z_out, c='tab:orange')
        ax.set_title(str(4*N_obs)+'-sampling of the necklace with '+str(N_out)+' outliers')
        ax.legend((plt_obs, plt_out), ('data', 'outliers'), loc='lower left')
    return data


if __name__ == "__main__":

    dataset_sampler = SwissRoll()
    name = '128_noise4_DTM'
    data, color = dataset_sampler.sample(128, noise = 0.4)
    
    st = DTMFiltration(data, m = 0.01,p = 10, dimension_max=1)
    st.persistence()
    pers_pairs = st.persistence_pairs()
    print(pers_pairs)
    pairings = np.array([[pers_pairs[0][1][0],pers_pairs[0][1][1]]])
    for pair in pers_pairs[1:-1]:
        pairings = np.vstack((pairings,np.array([[pair[1][0],pair[1][1]]])))

    make_plot(data, pairings, color, name = name)

    # name = '128_noise4_reg'
    # make_data(data, color, name = name)
    #
    # path_pairings = '{}pairings_{}.npy'.format(PATH_ROOT, name)
    # path_data = '{}data_{}.npy'.format(PATH_ROOT, name)
    # path_color = '{}color_{}.npy'.format(PATH_ROOT, name)
    # pairings, data, color = np.load(path_pairings), np.load(path_data), np.load(path_color)
    # print(type(pairings))
    # print(pairings.shape)
    # #
    # make_plot(data, pairings, color, name = name)