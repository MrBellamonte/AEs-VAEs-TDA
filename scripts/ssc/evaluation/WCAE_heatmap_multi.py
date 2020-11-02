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
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed_new_new.csv'
    save_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/'
    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)

    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ','test_mean_trustworthiness']
    metrics_pretty = [r'$MSE_{\matcal{M},\matcal{Z}}$', r'$\matcal{L}_r$', r'$\hat{\sigma}_{45}^{iso}$',r'$1-$Trust']
    max_metrics = ['test_mean_trustworthiness']




    j = 2
    js = [1,2,3]
    jbs = 1
    bss = [64,128,256,512]
    modes = ['mean', 'best']

    fig, ax = plt.subplots(nrows=len(bss),ncols= 2,figsize=(10,(5*len(bss))) )

    for j_mode,mode in enumerate(modes):
        for jbs,bs in enumerate(bss):



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

                df_temp = df_temp.rename(columns = {'value' : k, 'mu_push' : r'''$\nu$'''}).set_index([r'''$\nu$'''])

                print(df_temp)
                df_heatmap = df_heatmap.append(df_temp.T)

            print(df_heatmap)
            df_heatmap.index.name = 'number of neighbors'


            sns.heatmap(df_heatmap, cmap='coolwarm',robust = True, annot=True,cbar=False,ax=ax[jbs,j_mode])
            ax[jbs,j_mode].set_title(r'''$n_{bs}=${BS} ({mode})'''.format(bs = '{bs}', BS=bs,mode = mode))
    fig.tight_layout()
    plt.savefig(os.path.join(save_path,'heatmap_multi_{}.pdf'.format(metrics[j])))
    plt.show()
