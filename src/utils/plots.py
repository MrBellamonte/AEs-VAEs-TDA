import os
from collections import defaultdict
import random

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator


def plot_2Dscatter(data, labels, path_to_save= None, title = None, show = False, palette = plt.cm.viridis, dpi = 200):
    #fig, ax = plt.subplots()
    # sns_plot = sns.scatterplot(x = data[:, 0], y=data[:, 1], hue=labels, palette=palette, marker=".",
    #     #                 size=5, edgecolor="none", legend=True)
    if palette is 'hsv':
        sns_plot = plt.scatter(data[:, 0], data[:, 1],
                         c=labels, s=5, cmap="hsv")
        cbar = plt.colorbar(sns_plot)
        cbar.set_ticks([0,90,180,270,360])
        cbar.set_ticklabels(['0°','90°','180°','270°','360°'])

    else:
        sns_plot = sns.scatterplot(x = data[:, 0], y=data[:, 1], hue=labels, palette=palette, marker=".",
                                   size=5, edgecolor="none")
    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.title(title)

    if show:
        plt.show()

    if path_to_save != None:
        fig = sns_plot.get_figure()
        fig.savefig(path_to_save, dpi=dpi)

    plt.close()


def plot_losses(losses, losses_std=defaultdict(lambda: None), save_file=None, pairs_axes = False):
    """Plot a dictionary with per epoch losses.
    """
    palette = sns.color_palette()
    fig, ax = plt.subplots()
    if pairs_axes:
        ax2 = ax.twinx()
        ax2.set_ylim([0, 1])
    i = 0
    for key, values in losses.items():

        if ('matched_pairs_0D' in key and pairs_axes) or ('metrics.push_pairs' in key and pairs_axes):
            ax2.set_ylabel('matched_pairs_0D')

            #ax2.errorbar(range(len(values)), values, yerr=losses_std[key], label=key, color = palette[i])
            ax2.plot(range(len(values)), values,  label=key,color=palette[i], marker = '.')
            ax2.fill_between(range(len(values)), np.array(values)-np.array(losses_std[key]),  np.array(values)+np.array(losses_std[key]), alpha=.3, facecolor=palette[i])
        else:
            #ax.errorbar(range(len(values)), values, yerr=losses_std[key], label=key, color = palette[i])
            ax.plot(range(len(values)), values,  label=key,color=palette[i], marker = '.')
            ax.fill_between(range(len(values)), np.array(values)-np.array(losses_std[key]),  np.array(values)+np.array(losses_std[key]), alpha=.3, facecolor=palette[i])
        i += 1
    plt.xlabel('# epochs')
    ax.set_ylabel('loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if pairs_axes:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines+lines2, labels+labels2, loc=0)
        ax2.set_ylabel('matched pairs percentage')
    else:
        plt.legend()
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()


def plot_simplicial_complex_2D(simp_complex: list, points: np.ndarray, scale: float):
    """
    Util to plot simplicial complexes in 2D, for k-simplicies k=(0,1,2).

    :param simp_complex:
    :param points:
    :param scale:
    :return:
    """
    #todo: allow custom style?
    #todo: find a way that 2-simplex do not overlap (visually)

    #needed because over gudhi simp. complex object can only be iterated once for unknown reasons.
    complex = []
    for s in simp_complex:
        complex.append(s)


    # 2-simplicies
    triangles = np.array(
        [s[0] for s in complex if len(s[0]) == 3 and s[1] <= scale])
    triangle_patches = []
    temp2 = []
    for s in simp_complex:
        temp2.append(s)

    for idx in triangles:
        coord = np.column_stack(
            (points[idx, 0].reshape(3, ), points[idx, 1].reshape(3, )))
        polygon = Polygon(coord, True)
        triangle_patches.append(polygon)

    p = PatchCollection(triangle_patches, cmap=matplotlib.cm.jet, alpha=0.4)
    colors = 50*np.ones(len(triangle_patches))
    p.set_array(np.array(colors))

    plt.gca().add_collection(p)
    
    # 1-simplicies
    edges = np.array(
        [s[0] for s in complex if len(s[0]) == 2 and s[1] <= scale])
    for idx in edges:
        plt.plot(points[idx, 0], points[idx, 1], color='darkblue')

    # 0-simplicies
    plt.scatter(points[:, 0], points[:, 1], color='indigo', zorder=10)



def plot_distcomp_Z_manifold(Z_manifold, Z_latent, pwd_manifold, pwd_Z, labels, path_to_save = None,name = None,fontsize=24, show = False):
    Z_manifold[:, 0] = (Z_manifold[:,0]-Z_manifold[:,0].min())/(Z_manifold[:,0].max()-Z_manifold[:,0].min())
    Z_manifold[:, 1] = (Z_manifold[:,1]-Z_manifold[:,1].min())/(Z_manifold[:,1].max()-Z_manifold[:,1].min())
    Z_latent[:, 0] = (Z_latent[:,0]-Z_latent[:,0].min())/(Z_latent[:,0].max()-Z_latent[:,0].min())
    Z_latent[:, 1] = (Z_latent[:,1]-Z_latent[:,1].min())/(Z_latent[:,1].max()-Z_latent[:,1].min())

    latents = pd.DataFrame({'x': Z_latent[:, 0], 'y': Z_latent[:, 1],'label': labels})

    pwd_Z = pwd_Z
    pwd_Ztrue = pwd_manifold

    pwd_Ztrue = (pwd_Ztrue-pwd_Ztrue.min())/(pwd_Ztrue.max()-pwd_Ztrue.min())
    pwd_Z = (pwd_Z-pwd_Z.min())/(pwd_Z.max()-pwd_Z.min())

    #flatten
    pwd_Ztrue = pwd_Ztrue.flatten()
    pwd_Z = pwd_Z.flatten()

    if 2**12 <=len(pwd_Z):
        ind = random.sample(range(len(pwd_Z)), 2**12)
        pwd_Ztrue = pwd_Ztrue[ind]
        pwd_Z = pwd_Z[ind]

    distances = pd.DataFrame({'Distances on $\mathcal{M}$': pwd_Ztrue, 'Distances in $\mathcal{Z}$': pwd_Z})

    fig, ax = plt.subplots(2,1, figsize=(10, 20))

    sns.scatterplot(x = 'Distances on $\mathcal{M}$', y = 'Distances in $\mathcal{Z}$',data = distances, ax = ax[1], edgecolor = None,alpha=0.3)
    ax[1].xaxis.label.set_size(max(fontsize-2,12))
    ax[1].yaxis.label.set_size(max(fontsize-2,12))
    ax[1].set_title('Comparison of pairwise distances',fontsize=fontsize,pad=20)

    lims = [max(0, 0), min(1, 1)]
    ax[1].plot(lims, lims, '--',linewidth=5, color = 'black')

    sns.scatterplot(x = 'x', y = 'y',hue='label', data = latents,ax = ax[0],palette=plt.cm.viridis, marker=".", s=80,
                            edgecolor="none", legend=False)
    ax[0].set_title('Latent space ($\mathcal{Z}$)',fontsize=fontsize,pad=20)
    ax[0].set(xlabel="", ylabel="")
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    fig.tight_layout(pad=5)
    if path_to_save != None and name != None:
        print('save plot')
        fig.savefig(os.path.join(path_to_save,'{}.pdf'.format(name)),dpi = 100)
    if show:
        plt.show()
    plt.close()


def visualize_latents(latents, labels, save_file=None):
    plt.scatter(latents[:, 0], latents[:, 1], c=labels,
                cmap=plt.cm.Spectral, s=2., alpha=0.5)
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()

