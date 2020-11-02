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
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed_new.csv'
    save_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/contour'
    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)
    fig, axs = plt.subplots(ncols=4, figsize=(24,5),constrained_layout=True)
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ','test_mean_trustworthiness']
    metrics_pretty = [r'$MSE_{\matcal{M},\matcal{Z}}$', r'$\matcal{L}_r$', r'$\hat{\sigma}_{45}^{iso}$',r'$1-$Trust']
    max_metrics = ['rmse_manifold_Z']
    j = 0
    js = [1,2,3]
    jbs = 2
    bss = [64,128,256,512]



    modes = ['mean','best']
    mode = modes[1]
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 4),
                     axes_pad=0.25,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="10%",
                     cbar_pad=1,
                     )

    for jbs, bs in enumerate(bss):
        df_ = df[df['metric'] == metrics[j]]
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

        df_ = df_[df_['batch_size'] == bss[jbs]]
        df_ = df_.sort_values(by = 'mu_push')
        mu_pushs = list(set(list(df_.mu_push)))

        df_heatmap = pd.DataFrame()
        for k in sorted(set(list(df_.k)), reverse = True):
            df_temp = df_[df_.k == k][['mu_push', 'value']]

            df_temp = df_temp.set_index(['mu_push']).rename(columns = {'value' : k})


            #df_temp = df_temp[['mu_push','value']]



            #headers = df_temp.iloc[0]
            #new_df = pd.DataFrame(df_temp.values[1:], columns=headers)

            print(df_temp)
            df_heatmap = df_heatmap.append(df_temp.T)

        print(df_heatmap)

        #sns.heatmap(df_heatmap, cmap="YlGnBu", ax = axs[jbs])
        ax = axs[jbs]
        cntr = sns.heatmap(df_heatmap, cmap='viridis')

        make_square_axes(ax)
        if jbs == 2:
            cntr_ = cntr

    #grid[1].cax.colorbar(cntr_)
    grid[1].cax.toggle_label(True)
    grid[1].cax.tick_params(labelsize=15)
    grid[1].cax.set_label('a label')

    #grid[1].cax.set_title(metrics_pretty[j], fontsize=20, pad=20)
    plt.show()
