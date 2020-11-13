import os
from fractions import Fraction

import matplotlib.pyplot as plt
from matplotlib import cm
import mpltern
import pandas as pd
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid


def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

if __name__ == "__main__":
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed.csv'
    save_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/contour'
    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ','test_mean_trustworthiness']
    metrics_pretty = [r'$MSE_{\matcal{M},\matcal{Z}}$', r'$\matcal{L}_r$', r'$\hat{\sigma}_{45}^{iso}$',r'$1-$Trust']
    max_metrics = ['test_mean_trustworthiness']
    js = [1,2,3]
    bss = [64,128,256,512]
    fig, axs = plt.subplots(ncols=4, figsize=(24,5),constrained_layout=True)
    fig = plt.figure(figsize=(25,6))


    modes = ['mean','best']
    mode = modes[0]

    j = 3


    #fig.suptitle(metrics_pretty[j], fontsize=26)

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 4),
                     axes_pad=0.25,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="10%",
                     cbar_pad=1,
                     )

    #plt.tight_layout()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    vmin = np.zeros(len(js))
    vmax = np.ones(len(js)) * 1000
    # get vmin, vmax
    # for i in range(4):
    #     for j in range(len(js)):
    #         metric = metrics[js[j]]
    #         df_ = df[df['metric'] == metric]
    #         df_ = df_[['batch_size', 'mu_push', 'k', 'value','seed']]
    #
    #         if metric in max_metrics:
    #             df_ = df_.groupby(['batch_size', 'mu_push', 'k', 'seed'], as_index=False).max()
    #             df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
    #             df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()
    #
    #             df_['value'] = 1-df_['value']
    #         else:
    #             df_ = df_.groupby(['batch_size', 'mu_push', 'k','seed'], as_index=False).min()
    #             df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
    #             df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()
    #
    #         if vmin[j] > df_['value'].min():
    #             vmin[j] = df_['value'].min()
    #         else:
    #             pass
    #         if vmax[j] > df_['value'].max():
    #             vmax[j] = df_['value'].max()
    #         else:
    #             pass


    for i in range(4):

        df_ = df[df['metric'] == metrics[j]]
        df_ = df_[['batch_size', 'mu_push', 'k', 'value','seed']]

        if mode == 'mean':
            if metrics[j] in max_metrics:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k', 'seed'], as_index=False).max()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()

                df_['value'] = 1-df_['value']
            else:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k','seed'], as_index=False).min()
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).mean()
        else:
            if metrics[j] in max_metrics:
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).max()

                df_['value'] = 1-df_['value']
            else:
                df_ = df_[['batch_size', 'mu_push', 'k', 'value']]
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).min()

        ax = grid[i]
        ax.set_xlabel('k',fontsize=20)
        ax.set_ylabel(r'$\nu$', fontsize=20)
        ax.tick_params(labelsize=15)
        ax.set_title(r'$n_{bs}=$' + str(bss[i]),fontsize=22,pad=20)
        df_= df_[df_['batch_size'] == bss[i]]

        #ax.tricontour(df_['k'], df_['mu_push'], df_['value'], levels=14, linewidths=0.5, colors='k')

        cntr = ax.tricontourf(df_['k'], df_['mu_push'], df_['value'], levels=32, cmap=cm.get_cmap('viridis', 32))
        make_square_axes(ax)
        if i == 2:
            cntr_ = cntr


    grid[1].cax.colorbar(cntr_)
    grid[1].cax.toggle_label(True)
    grid[1].cax.tick_params(labelsize=15)
    grid[1].cax.set_label('a label')

    grid[1].cax.set_title(metrics_pretty[j], fontsize=20, pad=20)

    fig.savefig(os.path.join(save_path,'{}_{}.pdf'.format(mode,metrics[j].replace('.','_'))),pad_inches=0)
    fig.show()
