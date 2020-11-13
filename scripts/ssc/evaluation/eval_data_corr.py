import os

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def fancy_name(fancy_mapping: dict):
    return list(fancy_mapping.items())[0][1]


if __name__ == "__main__":

    exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise'
    df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed_new_new.csv'
    root_save = '/Users/simons/MT_data/eval_all_analysis/WCAE/SwissRoll_nonoise'


    criterion = 'rmse_manifold_Z'

    metrics = [
        'rmse_manifold_Z',
        'test_mean_Lipschitz_std_refZ',
        'test_mean_Lipschitz_std_refX',
        'test_mean_local_rmse_refX',
        'test_mean_local_rmse_refZ',
        'test_mean_trustworthiness',
        'test_mean_continuity',
        'test_density_kl_global_10',
        'test_density_kl_global_1',
        'test_density_kl_global_01',
        'test_density_kl_global_001',
        'test_density_kl_global_0001',
        'test_density_kl_global_00001',
        'test_density_kl_global_00001',
        'training.loss.autoencoder',
    ]

    max_metrics = ['test_mean_trustworthiness','test_mean_continuity']


    print('Load data...')
    df = pd.read_csv(df_path)

    df = df[df.batch_size == 256]


    print(set(df.batch_size.values))
    bss = [64, 128, 256, 512]

    df_criterion_metric = df[['uid','batch_size','metric','value','seed']]
    df_criterion_metric = df_criterion_metric[df_criterion_metric.metric == criterion]
    df_criterion_metric = df_criterion_metric[df_criterion_metric.batch_size == 256]

    df_selected = df_criterion_metric.sort_values('value', ascending=True).groupby(['seed','batch_size']).head(1)

    df = df[['uid','batch_size','metric','value']]
    uid_selected = list(df_selected.uid.values)

    df_data = df[df.uid.isin(uid_selected)]




    df = df[df['metric'].isin(metrics)]


    if True:
        y_label = 'normalized metric value'
        for metric in metrics:
            min = df.loc[df['metric'] == metric, 'value'].min()
            max = df.loc[df['metric'] == metric, 'value'].max()
            df.loc[df['metric'] == metric, 'value'] = (df.loc[
                                                           df['metric'] == metric, 'value']-min)/(
                                                              max-min)

    # SET X-VALUE CORRESPONDING TO METRIC


    # SET MARKER
    df['cat'] = 0
    df.loc[df['uid'].isin(uid_selected), 'cat'] = 1
    fig, ax = plt.subplots()
    sns.scatterplot(data=df[df['cat'] != 1], x="metric", y="value", color='black', ax = ax, label = 'rest')
    sns.scatterplot(data=df[df['cat'] == 1], x="metric", y="value", color='red',ax = ax, label = 'good example')
    #plt.xticks(np.arange(0, (len(metrics))), metrics,rotation='vertical')
    plt.xticks(rotation='vertical')
    plt.ylabel(y_label)
    plt.xlabel('')
    plt.show()

    #fig.savefig(os.path.join(path_to_save, '{}_{}.pdf'.format(name,name2)), dpi=200)
