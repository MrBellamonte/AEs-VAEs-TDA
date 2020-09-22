from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator


def plot_2Dscatter(data, labels, path_to_save= None, title = None, show = False):


    if len(np.unique(labels)) > 8:
        palette = "Spectral"
    else:
        palette  = "Dark2"


    sns_plot = sns.scatterplot(x = data[:, 0], y=data[:, 1], hue=labels, palette=sns.color_palette(palette, len(np.unique(labels))), marker=".",
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



