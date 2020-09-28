from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab


def plot_2Dscatter(data, labels, path_to_save=None, title=None, show=False):
    if len(np.unique(labels)) > 8:
        palette = "Spectral"
    else:
        palette = "Dark2"

    sns_plot = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels,
                               palette=sns.color_palette(palette, len(np.unique(labels))),
                               marker=".",
                               size=5, edgecolor="none", legend=False)
    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.title(title)

    if show:
        plt.show()

    if path_to_save != None:
        fig = sns_plot.get_figure()
        fig.savefig(path_to_save)

    plt.close()


def plot_latent_pretty_singlerow(latents:List[pd.DataFrame], cols: List[str], cmap=plt.cm.viridis, path_to_save=None,
                       show=False, name = None):
    fig, ax = plt.subplots(1, len(cols),
                           figsize=(len(cols)*7, 7))
    plt.axis('equal')
    # todo: move to bottom of row


    ax[1].set_title('',fontsize=25,pad=20)
    for col_i, latent_data in enumerate(latents):
        ax[col_i].set_title(cols[col_i], fontsize=25, pad=20)
        sns.scatterplot(x=latent_data['0'], y=latent_data['1'], hue=latent_data['labels'], palette=cmap, marker=".", s=30,
                            edgecolor="none", legend=False, ax = ax[col_i])

        if col_i < len(cols)-1:
            sns.despine(left=True, bottom=True, right=False, ax=ax[col_i])
        else:
            sns.despine(left=True, bottom=True, right=True, ax=ax[col_i])

        ax[col_i].tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False,
                            left=False)
        ax[col_i].axis('equal')
        ax[col_i].set_xticks([])
        ax[col_i].set_yticks([])
        ax[col_i].set_xticklabels([])
        ax[col_i].set_yticklabels([])
        ax[col_i].set_xlabel('')
        ax[col_i].set_ylabel('')


    if show:
        plt.show()
    if path_to_save != None and name != None:

        path_ = path_to_save + name + '.pdf'
        fig.savefig(path_, dpi = 400)


def plot_latent_pretty(latents: dict(), cols: List[str], cmap=plt.cm.viridis, path_to_save=None,
                       show=False):
    fig, ax = plt.subplots(len(list(latents.keys())), len(cols),
                           figsize=(len(cols)*7, (len(list(latents.keys()))*7)))
    plt.axis('equal')
    # todo: move to bottom of row
    rows = [key for key in list(latents.keys())]
    pad = 5  # in points

    for axx, col in zip(ax[0], cols):
        axx.annotate(col, xy=(0.5, 1), xytext=(0, 60),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=25, ha='center', va='baseline')

    for axx, row in zip(ax[:, 0], rows):
        axx.annotate(row, xy=(0, 0.5), xytext=(-axx.yaxis.labelpad-pad, 10),
                    xycoords=axx.yaxis.label, textcoords='offset points',
                    fontsize=20, ha='right', va='center')

    for row_i, key in enumerate(list(latents.keys())):
        ax[row_i, 1].set_title('',fontsize=25,pad=20)
        for col_i, latent_data in enumerate(latents[key]):

            sns.scatterplot(x=latent_data['0'], y=latent_data['1'], hue=latent_data['labels'], palette=cmap, marker=".", size=5,
                            edgecolor="none", legend=False, ax = ax[row_i,col_i])
            sns.despine(left=True, bottom=True,ax = ax[row_i,col_i])

            ax[row_i,col_i].tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False,
                            left=False)
            ax[row_i,col_i].axis('equal')

    pylab.plot([0, 1], [0.6, 0.5], color='red', lw=5, transform=pylab.gcf().transFigure,
               clip_on=False)
    pylab.plot([0.5, 0.5], [0, 1], color='lightgreen', lw=5, transform=pylab.gcf().transFigure,
               clip_on=False)
    if show:
        plt.show()


if __name__ == "__main__":
    # load latents (needs to be done manually...)
    test_latent_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCTopoAE_swissroll_apush_stability/SwissRoll-n_samples2560-Autoencoder_MLP_topoae-32-32-lr1_25-bs128-nep1000-rlw1-tlw8192-mepush_active5_4-k2-rmax10-seed588-5a66dcab/latents.csv'

    root_path = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/'

    topoae_suffix = ['TopoAE/bs64/latents.csv','TopoAE/bs128/latents.csv','TopoAE/bs256/latents.csv']
    wctopoae_suffix = ['WCTopoAE/bs64/latents.csv', 'WCTopoAE/bs128/latents.csv',
                     'WCTopoAE/bs256/latents.csv']

    topoae_paths = [(root_path + suffix) for suffix in topoae_suffix]
    wctopoae_pathss = [(root_path+suffix) for suffix in wctopoae_suffix]
    df_topoae = [pd.read_csv(path) for path in topoae_paths]
    df_wctopoae = [pd.read_csv(path) for path in wctopoae_pathss]

    cols = [r'$n_{bs} = 64$',r'$n_{bs} = 128$',r'$n_{bs} = 256$']


    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/latents_pretty/workshoppaper/'
    name_topoae = 'topoae_64-128-258_larger'
    name_wctopoae = 'wctopoae_64-128-258_larger'

    
    plot_latent_pretty_singlerow(df_wctopoae, cols = cols, show = True, path_to_save=path_to_save, name = None)
    plot_latent_pretty_singlerow(df_topoae, cols=cols, show=True, path_to_save=path_to_save,
                                 name=None)
    
