from fractions import Fraction

import matplotlib.pyplot as plt
from matplotlib import cm
import mpltern
import pandas as pd
import numpy as np


def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

if __name__ == "__main__":
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed.csv'

    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ','test_mean_trustworthiness']
    max_metrics = ['test_mean_trustworthiness']
    js = [1,2,3]
    bss = [64,128,256,512]
    fig, axs = plt.subplots(ncols=4,nrows=len(js))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    vmin = np.zeros(len(js))
    vmax = np.ones(len(js)) * 1000
    # get vmin, vmax
    for i in range(4):
        for j in range(len(js)):
            metric = metrics[js[j]]
            df_ = df[df['metric'] == metric]
            df_ = df_[['batch_size', 'mu_push', 'k', 'value','seed']]

            if metric in max_metrics:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k', 'seed'], as_index=False).max()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()

                df_['value'] = 1-df_['value']
            else:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k','seed'], as_index=False).min()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()

            if vmin[j] > df_['value'].min():
                vmin[j] = df_['value'].min()
            else:
                pass
            if vmax[j] > df_['value'].max():
                vmax[j] = df_['value'].max()
            else:
                pass



    for i in range(4):
        for j in range(len(js)):
            metric = metrics[js[j]]
            df_ = df[df['metric'] == metric]
            df_ = df_[['batch_size', 'mu_push', 'k', 'value','seed']]

            if metric in max_metrics:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k', 'seed'], as_index=False).max()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()

                df_['value'] = 1-df_['value']
            else:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k','seed'], as_index=False).min()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()

            ax = axs[j,i]
            df_= df_[df_['batch_size'] == bss[i]]

            #ax.tricontour(df_['k'], df_['mu_push'], df_['value'], levels=14, linewidths=0.5, colors='k')
            cntr = ax.tricontourf(df_['k'], df_['mu_push'], df_['value'], levels=32, cmap=cm.get_cmap('viridis', 32), vmax = vmax[j], vmin = vmin[j])
            make_square_axes(ax)
            if i == 3:
                #fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)
                fig.colorbar(cntr, ax=axs[j,:].ravel().tolist(), shrink=0.95,location='bottom')
                #fig.colorbar(cntr, ax=[axs[j,:]],location='bottom')
    fig.show()
