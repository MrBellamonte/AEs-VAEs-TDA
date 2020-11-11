import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.topology.witness_complex import WitnessComplex


def get_spiral(manifold_samples):
    #theta = np.radians(manifold_samples)

    r = manifold_samples**(0.4)/3
    x = r*np.cos(manifold_samples)
    y = r*np.sin(manifold_samples)
    return x,y

def plot_manifold(x_line,y_line,x_samples,y_samples, labels_samples,x_samples_witnesses = None,y_samples_witnesses = None,labels_samples_witnesses = None,pairings = [], show = False, path_to_save = None, name = 'default', show_manifold = True):
    if show_manifold:
        plt.plot(x_line, y_line,c = 'lightgrey',zorder=orderline)
    sns.scatterplot(x=x_samples, y=y_samples, hue=labels_samples, palette=plt.cm.viridis, s=100,legend=False, zorder=orderscatter,linewidth = 0)
    if x_samples_witnesses is not None:
        sns.scatterplot(x=x_samples_witnesses, y=y_samples_witnesses, hue=labels_samples_witnesses, palette=plt.cm.viridis, s=100,
                        legend=False, zorder=orderscatter, alpha = 0.2,linewidth = 0)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)


    for pairing in pairings:
        plt.plot([x_samples[pairing[0]], x_samples[pairing[1]]],
                [y_samples[pairing[0]], y_samples[pairing[1]]], color='black',zorder=oder_graph)


    if path_to_save != None and name != None:
        print('save plot')
        plt.savefig(os.path.join(path_to_save,'{}.pdf'.format(name)),dpi = 100)
    if show:
        plt.show()
    plt.close()


def get_kNN_torch(X,dist = False):
    if dist:
        X_dist = torch.from_numpy(X)
    else:
        X = torch.from_numpy(X)
        X_dist = torch.norm(X[:, None]-X, dim=2, p=2)
    sorted, indices = torch.sort(X_dist)

    #kNN_mask = torch.zeros((X.size(0), X.size(0),)).scatter(1, indices[:, 1:(k+1)],1)
    return np.array(indices)

def collect_kNN(indices, k=1):
    pairs = []
    for ind in indices:
        for j in ind[1:(k+1)]:
            pairs.append([ind[0],j])
    return pairs

if __name__ == "__main__":



    # SPIRAL
    n_samples_many = 64
    n_samples_sparse = 32
    n_grid = 500

    min = 0
    max = 380+180
    orderline = 3
    oder_graph = 4
    orderscatter = 5

    K = 1
    seed =12

    path_to_save = '/Users/simons/polybox/Studium/20FS/MT/Presentations/visualizations/Spiral/k{}_s{}'.format(K,seed)
    try:
        os.mkdir(path_to_save)
    except:
        pass
    # sample from line
    np.random.seed(seed)
    random.seed(seed)
    samples_many = np.radians(np.random.uniform(0, max, n_samples_many))
    ind_sparse = random.sample(range(n_samples_many),n_samples_sparse)
    ind_witness = list(set(list(range(n_samples_many)))-set(ind_sparse))
    samples_sparse = samples_many[ind_sparse]


    line = np.radians(np.linspace(0, max, n_grid))
    x_line, y_line = get_spiral(line)


    # many
    x_samples_many, y_samples_many = get_spiral(samples_many)

    plot_manifold(line,np.zeros_like(line),samples_many,np.zeros_like(samples_many),labels_samples=samples_many, path_to_save=path_to_save,name = 'spiral_manifold_many{}{}'.format(K,seed), show = False)
    plot_manifold(x_line,y_line,x_samples_many,y_samples_many,labels_samples=samples_many,path_to_save=path_to_save,name = 'spiral_data_many{}{}'.format(K,seed), show = False)

    # sparse
    x_samples_sparse, y_samples_sparse = get_spiral(samples_sparse)
    plot_manifold(line,np.zeros_like(line),samples_sparse,np.zeros_like(samples_sparse),labels_samples=samples_sparse, path_to_save=path_to_save,name = 'spiral_manifold_sparse{}{}'.format(K,seed), show = False)
    plot_manifold(x_line,y_line,x_samples_sparse,y_samples_sparse,labels_samples=samples_sparse,path_to_save=path_to_save,name = 'spiral_data_sparse{}{}'.format(K,seed), show = False)

    # many graph
    x_samples_many, y_samples_many = get_spiral(samples_many)
    X_manifold_many = np.concatenate((samples_many.reshape(len(samples_many),1), np.zeros_like(samples_many).reshape(len(samples_many),1)),axis=1)
    X_R2_many = np.concatenate((samples_many.reshape(len(x_samples_many),1), y_samples_many.reshape(len(samples_many),1)),axis=1)


    knn_many_manifold = get_kNN_torch(X_manifold_many)
    pairings_many_manifold = collect_kNN(knn_many_manifold,k=K)

    knn_many_R2 = get_kNN_torch(X_R2_many)
    pairings_many_R2 = collect_kNN(knn_many_R2,k=K)


    plot_manifold(line, np.zeros_like(line), samples_many, np.zeros_like(samples_many),labels_samples=samples_many,
                  path_to_save=path_to_save, name='spiral_manifold_many_wgraph{}{}'.format(K,seed), show=False,pairings=pairings_many_manifold, show_manifold=False)
    plot_manifold(x_line, y_line, x_samples_many, y_samples_many, labels_samples=samples_many, path_to_save=path_to_save,
                  name='spiral_data_many_wgraph{}{}'.format(K,seed), show=False,pairings=pairings_many_R2, show_manifold=False)

    # sparse graph
    x_samples_sparse, y_samples_sparse = get_spiral(samples_sparse)
    X_manifold_sparse = np.concatenate((samples_sparse.reshape(len(samples_sparse),1), np.zeros_like(samples_sparse).reshape(len(samples_sparse),1)),axis=1)

    X_R2_sparse = np.concatenate((x_samples_sparse.reshape(len(x_samples_sparse),1), y_samples_sparse.reshape(len(x_samples_sparse),1)),axis=1)

    knn_sparse_manifold = get_kNN_torch(X_manifold_sparse)
    pairings_sparse_manifold = collect_kNN(knn_sparse_manifold,k=K)

    knn_sparse_R2 = get_kNN_torch(X_R2_sparse)
    pairings_sparse_R2 = collect_kNN(knn_sparse_R2,k=K)

    plot_manifold(line, np.zeros_like(line), samples_sparse, np.zeros_like(samples_sparse),labels_samples=samples_sparse,
                  path_to_save=path_to_save, pairings=pairings_sparse_manifold,name='spiral_manifold_sparse_wgraph{}{}'.format(K,seed), show=False, show_manifold=False)
    plot_manifold(x_line, y_line, x_samples_sparse, y_samples_sparse,labels_samples=samples_sparse,pairings=pairings_sparse_R2, path_to_save=path_to_save,
                  name='spiral_data_sparse_wgraph{}{}'.format(K,seed), show=True, show_manifold=False)




    witnesses_tensor = torch.from_numpy(X_manifold_many)
    landmarks_tensor = torch.from_numpy(X_manifold_sparse)

    witness_complex = WitnessComplex(landmarks_tensor, witnesses_tensor)
    witness_complex.compute_metric_optimized()
    dist_wc = witness_complex.landmarks_dist

    knn_sparse_manifold_wc = get_kNN_torch(dist_wc.numpy(),dist = True)
    pairings_sparse_R2_wd = collect_kNN(knn_sparse_manifold_wc, k=K)

    plot_manifold(x_line, y_line, x_samples_sparse, y_samples_sparse,labels_samples=samples_sparse,x_samples_witnesses = x_samples_many[ind_witness], y_samples_witnesses = y_samples_many[ind_witness],pairings=pairings_sparse_R2_wd, path_to_save=path_to_save,
                  name='spiral_data_sparse_wgraph_witnesscomplex{}{}'.format(K,seed), show=True, show_manifold=False, labels_samples_witnesses=samples_many[ind_witness])
