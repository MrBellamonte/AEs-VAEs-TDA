from fractions import Fraction

import matplotlib.pyplot as plt
import mpltern
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed.csv'

    # get df with cols: eval metrics (tbd), uid, k, bs, mu_push
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_Lipschitz_std_refZ']

    # get uids, get mu_push, k out of uid

    df = pd.read_csv(df_path)
    metrics = ['rmse_manifold_Z', 'training.loss.autoencoder', 'test_mean_Lipschitz_std_refZ',
               'test_mean_trustworthiness']
    max_metrics = ['test_mean_trustworthiness']
    js = [1,2,3]
    bss = [64,128,256,512]
    fig, axs = plt.subplots(ncols=4,nrows=len(js),figsize=(20, len(js)*5))

    # metric = metrics[2]
    # df = df[df['metric'] == metric]
    # df = df[['batch_size','mu_push', 'k', 'value']]
    #
    # if metric in max_metrics:
    #     df = df.groupby(['batch_size','mu_push', 'k'], as_index=False).min()
    # else:
    #     df = df.groupby(['batch_size','mu_push', 'k'], as_index=False).min()

    for i in range(4):
        for j in range(len(js)):
            metric = metrics[js[j]]
            df_ = df[df['metric'] == metric]
            df_ = df_[['batch_size', 'mu_push', 'k', 'value']]

            if metric in max_metrics:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).max()
            else:
                df_ = df_.groupby(['batch_size', 'mu_push', 'k'], as_index=False).min()

            ax = axs[j,i]
            df_= df_[df_['batch_size'] == bss[i]]

            #ax.tricontour(df_['k'], df_['mu_push'], df_['value'], levels=14, linewidths=0.5, colors='k')
            cntr = ax.tricontourf(df_['k'], df_['mu_push'], df_['value'], levels=100, cmap="RdBu_r")
            fig.colorbar(cntr, ax=ax)
    fig.show()
