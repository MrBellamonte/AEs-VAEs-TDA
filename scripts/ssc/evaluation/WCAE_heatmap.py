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
    df_path = '/Users/simons/MT_data/eval_data/SWISSROLL_FINAL/WCAE/eval_metrics_all_wkmu.csv'
    save_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/'
    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)

    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ','test_mean_trustworthiness']
    metrics_pretty = [r'$MSE_{\matcal{M},\matcal{Z}}$', r'$\matcal{L}_r$', r'$\hat{\sigma}_{45}^{iso}$',r'$1-$Trust']
    max_metrics = ['test_mean_trustworthiness']
    j = 0
    js = [1,2,3]
    jbs = 1
    bss = [64,128,256,512]
    modes = ['mean', 'best']
    for mode in modes:
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


                #df_temp = df_temp[['mu_push','value']]



                #headers = df_temp.iloc[0]
                #new_df = pd.DataFrame(df_temp.values[1:], columns=headers)

                print(df_temp)
                df_heatmap = df_heatmap.append(df_temp.T)

            print(df_heatmap)
            df_heatmap.index.name = 'number of neighbors'


            sns.heatmap(df_heatmap, cmap='coolwarm',robust = True, annot=True,cbar=False,square=True)
            plt.savefig(os.path.join(save_path,'heatmap_bs{}_j{}_mode{}.pdf'.format(bss[jbs],metrics[j],mode)))
            plt.show()
