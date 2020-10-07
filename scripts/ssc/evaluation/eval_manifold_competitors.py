import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from scripts.ssc.evaluation.mldl_copied import CompPerformMetrics
from src.datasets.datasets import SwissRoll, SwissRoll_manifold
from src.evaluation.eval import Multi_Evaluation
from src.models.COREL.eval_engine import get_latentspace_representation
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae


def update_dict(dict, ks, metric, result):
    for i, k in enumerate(ks):
        dict.update({metric+'_k{}'.format(k): result[metric][i]})

    return dict

def plot_dist_comparison(Z_manifold, Z_latent, labels, path_to_save = None,name = None):

    print('normalize x,y')
    Z_manifold[:, 0] = (Z_manifold[:,0]-Z_manifold[:,0].min())/(Z_manifold[:,0].max()-Z_manifold[:,0].min())
    Z_manifold[:, 1] = (Z_manifold[:,1]-Z_manifold[:,1].min())/(Z_manifold[:,1].max()-Z_manifold[:,1].min())
    Z_latent[:, 0] = (Z_latent[:,0]-Z_latent[:,0].min())/(Z_latent[:,0].max()-Z_latent[:,0].min())
    Z_latent[:, 1] = (Z_latent[:,1]-Z_latent[:,1].min())/(Z_latent[:,1].max()-Z_latent[:,1].min())

    manifold = pd.DataFrame({'x': Z_manifold[:, 0], 'y': Z_manifold[:, 1],'label': labels})
    latents = pd.DataFrame({'x': Z_latent[:, 0], 'y': Z_latent[:, 1],'label': labels})

    print('compute distances')
    pwd_Z = pairwise_distances(Z_eval, Z_eval, n_jobs=2)
    pwd_Ztrue = pairwise_distances(data_manifold, data_manifold, n_jobs=2)
    print('normalize distances')
    #normalize distances
    pwd_Ztrue = (pwd_Ztrue-pwd_Ztrue.min())/(pwd_Ztrue.max()-pwd_Ztrue.min())
    pwd_Z = (pwd_Z-pwd_Z.min())/(pwd_Z.max()-pwd_Z.min())

    print('flatten')
    #flatten
    pwd_Ztrue = pwd_Ztrue.flatten()
    pwd_Z = pwd_Z.flatten()

    ind = random.sample(range(len(pwd_Z)), 2**12)


    distances = pd.DataFrame({'Distances on $\mathcal{M}$': pwd_Ztrue[ind], 'Distances in $\mathcal{Z}$': pwd_Z[ind]})

    print('plot')
    #plot
    fig, ax = plt.subplots(1,3, figsize=(3*10, 10))

    sns.scatterplot(x = 'Distances on $\mathcal{M}$', y = 'Distances in $\mathcal{Z}$',data = distances, ax = ax[1], edgecolor = None,alpha=0.3)
    #ax[0].set(xlabel='Distances on $\mathcal{M}$', ylabel='Distances in $\mathcal{Z}$',fontsize=25)
    ax[1].xaxis.label.set_size(20)
    ax[1].yaxis.label.set_size(20)
    ax[1].set_title('Comparison of pairwise distances',fontsize=24,pad=20)


    sns.scatterplot(y = 'x', x = 'y', hue='label', data = manifold,ax = ax[0],palette=plt.cm.viridis, marker=".", s=80,
                            edgecolor="none", legend=False)
    ax[0].set_title('True manifold ($\mathcal{M}$)',fontsize=24,pad=20)
    ax[0].set(xlabel="", ylabel="")
    ax[0].set_yticks([])

    sns.scatterplot(x = 'x', y = 'y',hue='label', data = latents,ax = ax[2],palette=plt.cm.viridis, marker=".", s=80,
                            edgecolor="none", legend=False)
    ax[2].set_title('Latent space ($\mathcal{Z}$)',fontsize=24,pad=20)
    ax[2].set(xlabel="", ylabel="")
    ax[2].set_yticks([])
    fig.tight_layout(pad=5)
    if path_to_save != None and name != None:
        print('save plot')
        fig.savefig(os.path.join(path_to_save,'{}_4.pdf'.format(name)),dpi = 100)

    plt.show()
    plt.close()

    return (np.square(pwd_Ztrue - pwd_Z)).mean()


def plot_dist_comparison2(Z_manifold, Z_latent, labels, path_to_save = None,name = None):

    print('normalize x,y')
    Z_manifold[:, 0] = (Z_manifold[:,0]-Z_manifold[:,0].min())/(Z_manifold[:,0].max()-Z_manifold[:,0].min())
    Z_manifold[:, 1] = (Z_manifold[:,1]-Z_manifold[:,1].min())/(Z_manifold[:,1].max()-Z_manifold[:,1].min())
    Z_latent[:, 0] = (Z_latent[:,0]-Z_latent[:,0].min())/(Z_latent[:,0].max()-Z_latent[:,0].min())
    Z_latent[:, 1] = (Z_latent[:,1]-Z_latent[:,1].min())/(Z_latent[:,1].max()-Z_latent[:,1].min())

    manifold = pd.DataFrame({'x': Z_manifold[:, 0], 'y': Z_manifold[:, 1],'label': labels})
    latents = pd.DataFrame({'x': Z_latent[:, 0], 'y': Z_latent[:, 1],'label': labels})

    print('compute distances')
    pwd_Z = pairwise_distances(Z_eval, Z_eval, n_jobs=2)
    pwd_Ztrue = pairwise_distances(data_manifold, data_manifold, n_jobs=2)
    print('normalize distances')
    #normalize distances
    pwd_Ztrue = (pwd_Ztrue-pwd_Ztrue.min())/(pwd_Ztrue.max()-pwd_Ztrue.min())
    pwd_Z = (pwd_Z-pwd_Z.min())/(pwd_Z.max()-pwd_Z.min())

    print('flatten')
    #flatten
    pwd_Ztrue = pwd_Ztrue.flatten()
    pwd_Z = pwd_Z.flatten()

    ind = random.sample(range(len(pwd_Z)), 2**12)


    distances = pd.DataFrame({'Distances on $\mathcal{M}$': pwd_Ztrue[ind], 'Distances in $\mathcal{Z}$': pwd_Z[ind]})

    print('plot')
    #plot
    fig, ax = plt.subplots(2,1, figsize=(10, 20))

    sns.scatterplot(x = 'Distances on $\mathcal{M}$', y = 'Distances in $\mathcal{Z}$',data = distances, ax = ax[1], edgecolor = None,alpha=0.3)
    #ax[0].set(xlabel='Distances on $\mathcal{M}$', ylabel='Distances in $\mathcal{Z}$',fontsize=25)
    ax[1].xaxis.label.set_size(20)
    ax[1].yaxis.label.set_size(20)
    ax[1].set_title('Comparison of pairwise distances',fontsize=24,pad=20)

    lims = [max(0, 0), min(1, 1)]
    ax[1].plot(lims, lims, '--',linewidth=5, color = 'black')

    sns.scatterplot(x = 'x', y = 'y',hue='label', data = latents,ax = ax[0],palette=plt.cm.viridis, marker=".", s=80,
                            edgecolor="none", legend=False)
    ax[0].set_title('Latent space ($\mathcal{Z}$)',fontsize=24,pad=20)
    ax[0].set(xlabel="", ylabel="")
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    fig.tight_layout(pad=5)
    if path_to_save != None and name != None:
        print('save plot')
        fig.savefig(os.path.join(path_to_save,'{}_5.pdf'.format(name)),dpi = 100)

    plt.show()
    plt.close()

    return (np.square(pwd_Ztrue - pwd_Z)).mean()

if __name__ == "__main__":

    # create df and set path to save
    df_tot = pd.DataFrame()
    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/SwissRoll_manifold_comparison_competitors' \
                   ''

    # set which models to evaluate
    UMAP_seed = 887
    UMAP_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/umap_swissroll_2/SwissRoll-n_samples2560-UMAP--n_neighbors32-min_dist1_2-seed887-3db9b673/train_latents.npz'

    tSNE_see = 672
    tSNE_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/tsne_swissroll_2/SwissRoll-n_samples2560-tSNE--n_jobs1-perplexity50-seed672-017a5cba/train_latents.npz'

    eval_models_dict = {
        'UMAP': UMAP_path,
        'tSNE': tSNE_path
    }
    eval_seeds = {
        'UMAP': UMAP_seed,
        'tSNE': tSNE_see
    }







    model_names = []
    values = []
    for model_name, path in eval_models_dict.items():
        # load WC-AE
        # sample data
        # sample data
        n_samples = 2560
        manifold = SwissRoll_manifold()
        X_eval, data, y_eval = manifold.sample_all(n_samples=n_samples, seed=eval_seeds[model_name])

        X_eval, _, y_eval, __, = train_test_split(X_eval, y_eval,
                                                           test_size=0.2, random_state=eval_seeds[model_name])



        Z_eval = np.load(path)['latents']
        y_eval2 = np.load(path)['labels']
        print(y_eval2 == y_eval)


        value = plot_dist_comparison2(X_eval, Z_eval, y_eval, path_to_save=path_to_save,name = model_name)

        print('{}: {}'.format(model_name,value))

        rows = dict(
            model = model_name, rmse = value
        )

        df = pd.DataFrame({k: [v] for k, v in rows.items()})
        df_tot = df_tot.append(df)


    df_tot.to_csv(os.path.join(path_to_save,'rmse_data.csv'))