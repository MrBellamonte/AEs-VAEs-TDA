import os

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from scripts.ssc.pairings_visualization.utils_definitions import make_plot
from src.competitors.competitor_models import UMAP
from src.datasets.datasets import MNIST_offline

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_plot(data,pairings,labels,show=False,path_to_save=None):
    fig, ax = plt.subplots()
    for i,pairing in enumerate(pairings):
        for ind in pairing:
            ax.plot([data[i, 0], data[ind, 0]],
                    [data[i, 1], data[ind, 1]], color='grey',zorder = 1)
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=sns.color_palette("hls", 10),
                               size=100, edgecolor="none",ax = ax,zorder = 2)

    handles, labels_ = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels_, handles = zip(*sorted(zip(labels_, handles), key=lambda t: int(t[0])))

    if labels_[-1]=='100':
        ax.legend(handles[:-1], labels_[:-1])
    else:
        ax.legend(handles, labels_)

    ax.set(xlabel="", ylabel="")
    ax.set_yticks([])
    ax.set_xticks([])
    sns.despine(left=True, bottom=True)

    if show:
        plt.show()

    if path_to_save != None:

        fig.savefig(path_to_save, dpi=200)

    plt.close()


if __name__ == "__main__":

    root_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs128-seed838-noiseNone-4f608157'

    root_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/mnist_umap_nncomp/test'
    # load data batchwise into df, create integer id + batch id
    dataloader = torch.load(os.path.join(root_to_data,'dataloader_train.pt'))
    landmark_dist = torch.load(os.path.join(root_to_data, 'landmark_dist_train.pt'))
    euc_dist = torch.load(os.path.join(root_to_data, 'dist_X_all_train.pt'))

    Xs = []
    ys = []
    for i, (X_batch, label_batch) in enumerate(dataloader):
        if i == 0:
            X0 = X_batch
        Xs.append(X_batch)
        ys.append(label_batch)


    data = torch.cat(Xs,dim = 0)
    labels = torch.cat(ys, dim=0)



    model = UMAP()
    data_l, labels_l = model.get_latent_train(data.numpy(),labels.numpy())

    print(torch.all(torch.from_numpy(labels_l).eq(labels)))
    bs = 128
    ks = [1,2,3,4]
    for batch_i, (X_batch, label_batch) in enumerate(dataloader):

        #X,y = model.get_latent_test(X_batch.numpy(), label_batch.numpy())
        X,y = data_l[(bs*batch_i):bs*(batch_i+1),:],labels_l[(bs*batch_i):bs*(batch_i+1)]
        print(X.shape)
        print(y.shape)
        euc_dist_bi = euc_dist[batch_i,:,:].numpy()
        landmark_dist_bi = landmark_dist[batch_i, :, :].numpy()

        for k in ks:
            neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(landmark_dist_bi)
            distances, pairings = neigh.kneighbors(landmark_dist_bi)

            make_plot(X,pairings,y.astype(int).astype(str),True,path_to_save=os.path.join(root_to_save,'{}nnwc_bs128_4f608157_{}.pdf'.format(k,batch_i)))

            neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(euc_dist_bi)
            distances, pairings = neigh.kneighbors(euc_dist_bi)

            make_plot(X, pairings, y.astype(int).astype(str), True,
                      path_to_save=os.path.join(root_to_save,
                                                '{}nn_bs128_4f608157_{}.pdf'.format(k, batch_i)))

        if batch_i==1:
            break

    pass