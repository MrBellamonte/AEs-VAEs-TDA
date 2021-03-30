import os
import random

import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scripts.ssc.pairings_visualization.utils_definitions import make_plot
from src.competitors.competitor_models import UMAP
from src.datasets.datasets import MNIST_offline

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def count_out_of_class(pairings, labels):
    count = 0
    for i, pairing in enumerate(pairings):
        for ind in pairing:

            if labels[i] != labels[ind]:
                count = count + 1
            else:
                pass
    return count


if __name__ == "__main__":

    df = pd.DataFrame()

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




        ks = [1,2,3,4]
        maxk = 4
        count_nn = {
            1 : [], 2 : [], 3 : [], 4 : []
        }
        count_nnwc = {
            1 : [], 2 : [], 3 : [], 4 : []
        }

        for batch_i, (X_batch, label_batch) in enumerate(dataloader):
            if True:
                print('compute')
                try:
                    os.mkdir(os.path.join(root_to_save,'{}'.format(batch_i)))
                except:
                    pass
                #X,y = model.get_latent_test(X_batch.numpy(), label_batch.numpy())
                #y = labels.numpy()[(bs*batch_i):bs*(batch_i+1)]
                y = label_batch.numpy()
                euc_dist_bi = euc_dist[batch_i,:,:].numpy()
                landmark_dist_bi = landmark_dist[batch_i, :, :].numpy()

                for k in ks:
                    neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(landmark_dist_bi)
                    distances, pairings = neigh.kneighbors(landmark_dist_bi)

                    count_nnwc[k].append(count_out_of_class(pairings, y.astype(int).astype(str)))

                    neigh = NearestNeighbors(n_neighbors=(1+k), metric='precomputed').fit(euc_dist_bi)
                    distances, pairings = neigh.kneighbors(euc_dist_bi)

                    count_nn[k].append(count_out_of_class(pairings, y.astype(int).astype(str)))
            else:
                pass

        df_temp = pd.DataFrame(index=np.arange(8), columns=['method','bs','k','count'])

        for k in ks:
            df_temp['method'][(k-1)] = 'nn'
            df_temp['method'][(maxk+k-1)]  = 'nnwc'
            df_temp['k'][(k-1)] = k
            df_temp['k'][(maxk+k-1)]  = k
            df_temp['bs'][(k-1)] = bs
            df_temp['bs'][(maxk+k-1)]  = bs
            df_temp['count'][(k-1)] = np.array(count_nn[k]).mean()
            df_temp['count'][(maxk+k-1)]  = np.array(count_nnwc[k]).mean()

        df = df.append(df_temp)
    
    df.to_csv('/Users/simons/PycharmProjects/MT-VAEs-TDA/output/mnist_umap_nncomp/count.csv')

