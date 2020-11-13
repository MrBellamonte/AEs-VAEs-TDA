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

    metric = metrics[2]

    df = df[df['metric'] == metric]
    df = df[['batch_size', 'mu_push', 'k', 'value']]
    if metric in max_metrics:
        df = df.groupby(['batch_size', 'mu_push', 'k'], as_index=False).min()
    else:
        df = df.groupby(['batch_size', 'mu_push', 'k'], as_index=False).min()
    print(df)
    ax = plt.subplot(projection='ternary')

    bs_replace = {
        64 : 0,
        128: 0.25,
        256: 0.5,
        512: 1
    }

    k_replace = {
        1: 0,
        2: 0.2,
        3: 0.4,
        4: 0.6,
        5: 0.8,
        6: 1,
    }

    mu_replace = {
        1   : 0,
        1.05: 0.2,
        1.1 : 0.4,
        1.15: 0.6,
        1.2 : 0.8,
        1.25: 1,
    }

    bs = list(df['batch_size'].replace(bs_replace))
    k = list(df['k'].replace(k_replace))
    mu = list(df['mu_push'].replace(mu_replace))
    v = list(df['value'])
    print(df)
    vmin = 0.0
    vmax = 1.2
    levels = np.linspace(vmin, vmax, 7)

    cs = ax.tricontourf(bs, k, mu, v)
    #cs = ax.tripcolor(bs, k, mu, v, shading = 'gouraud', rasterized = True)
    #cs = ax.tripcolor(bs, k, mu, v, shading='flat')
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9])
    colorbar = plt.colorbar(cs, cax=cax)
    colorbar.set_label('Length', rotation=270, va='baseline')

    ax.set_tlabel('Batch size')
    ax.set_llabel('k')
    ax.set_rlabel('mu')
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')
    plt.show()
