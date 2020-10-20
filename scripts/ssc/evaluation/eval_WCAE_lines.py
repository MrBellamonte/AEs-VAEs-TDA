import os
from fractions import Fraction

import matplotlib.pyplot as plt
import seaborn as sns
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

    fig, axs = plt.subplots(nrows=4)


    modes = ['mean','best']
    mode = modes[0]

    j = 3

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

        ax = axs[i]

        ax.tick_params(labelsize=15)
        ax.set_title(r'$n_{bs}=$' + str(bss[i]),fontsize=22,pad=20)
        df_= df_[df_['batch_size'] == bss[i]]

        sns.lineplot(x="mu_push", y="value",
                     hue="k",
                     data=df_, ax=ax)
        ax.set_ylabel('k',fontsize=20)
        ax.set_xlabel(r'$\nu$', fontsize=20)











            #ax.tricontour(df_['k'], df_['mu_push'], df_['value'], levels=14, linewidths=0.5, colors='k')
            #cntr = ax.tricontourf(df_['k'], df_['mu_push'], df_['value'], levels=32, cmap=cm.get_cmap('viridis', 32), vmax = vmax[j], vmin = vmin[j])
            #make_square_axes(ax)

fig.show()
