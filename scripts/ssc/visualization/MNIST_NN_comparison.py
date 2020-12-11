import os
import random

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
    cp = dict()
    for i, c in enumerate(sns.color_palette("hls", 10)):
        cp.update({str(i):c})
    if pairings is None:
        pass
    else:
        x_pairs = []
        y_pairs = []
        label = pairs = []
        for i,pairing in enumerate(pairings):
            for ind in pairing:



                if labels[i] != labels[ind]:
                    ax.plot([data[i, 0], data[ind, 0]],
                            [data[i, 1], data[ind, 1]], color='grey',zorder = 1)
                else:
                    ax.plot([data[i, 0], data[ind, 0]],
                            [data[i, 1], data[ind, 1]], color=cp[labels[ind]], zorder=1)
                    #sns.lineplot(x=[data[i, 0], data[ind, 0]], y=[data[i, 1], data[ind, 1]], ax=ax, label=labels[ind],palette=[cp[labels[ind]]],zorder = 1)

    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=cp,
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

    BATCHES = [6,31,89,1001]

    NN = 10
    m_dist = 1
    m_dist_str = '1'
    trial = 1
    bs = 128
    for bs in [64,128,256,512,1014]:
        if bs == 64:
            folder = 'MNIST_offline-bs64-seed838-noiseNone-20738678'
        elif bs == 128:
            folder = 'MNIST_offline-bs128-seed838-noiseNone-4f608157'
        elif bs == 256:
            folder = 'MNIST_offline-bs256-seed838-noiseNone-4a5487de'
        elif bs == 512:
            folder = 'MNIST_offline-bs512-seed838-noiseNone-ced06774'
        elif bs == 1024:
            folder = 'MNIST_offline-bs1024-seed838-noiseNone-6f31dea2'
        else:
            ValueError

        root_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/{}'.format(folder)

        root_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/mnist_umap_nncomp/bs{}NN{}md{}tr{}'.format(bs,NN,m_dist_str,trial)
        try:
            os.mkdir(root_to_save)
        except:
            pass
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



        model = UMAP(n_neighbors = NN, min_dist = m_dist)
        data_l, labels_l = model.get_latent_train(data.numpy(),labels.numpy())
        ind_plot = random.sample(range(data.shape[0]), 15000)
        make_plot(data_l[ind_plot,:],None, labels_l[ind_plot].astype(int).astype(str), True, path_to_save=os.path.join(root_to_save,
                                                                                          'all_base.pdf'))
        print(torch.all(torch.from_numpy(labels_l).eq(labels)))

        ks = [1,2,3,4]
        for batch_i, (X_batch, label_batch) in enumerate(dataloader):
            if batch_i == 7 or batch_i == 33 or batch_i == 66:
                print('compute')
                try:
                    os.mkdir(os.path.join(root_to_save,'{}'.format(batch_i)))
                except:
                    pass
                #X,y = model.get_latent_test(X_batch.numpy(), label_batch.numpy())
                X,y = data_l[(bs*batch_i):bs*(batch_i+1),:],labels_l[(bs*batch_i):bs*(batch_i+1)]

                euc_dist_bi = euc_dist[batch_i,:,:].numpy()
                landmark_dist_bi = landmark_dist[batch_i, :, :].numpy()

                for k in ks:
                    neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(landmark_dist_bi)
                    distances, pairings = neigh.kneighbors(landmark_dist_bi)

                    make_plot(X,pairings,y.astype(int).astype(str),True,path_to_save=os.path.join(root_to_save,'{}/{}_nnwc_bs128_4f608157.pdf'.format(batch_i, k)))

                    neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(euc_dist_bi)
                    distances, pairings = neigh.kneighbors(euc_dist_bi)

                    make_plot(X, pairings, y.astype(int).astype(str), True,
                              path_to_save=os.path.join(root_to_save,
                                                        '{}/{}_nn_bs128_4f608157.pdf'.format(batch_i, k)))
            else:
                pass

