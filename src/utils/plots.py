from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator


def plot_classes_qual(data, labels, path_to_save= None, title = None, show = False):


    if len(np.unique(labels)) > 8:
        palette = "Spectral"
    else:
        palette  = "Dark2"


    sns_plot = sns.scatterplot(data[:, 0], data[:, 1], hue=labels, palette=sns.color_palette(palette, len(np.unique(labels))), marker=".",
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


def plot_losses(losses, losses_std=defaultdict(lambda: None), save_file=None):
    """Plot a dictionary with per epoch losses.


    """

    fig, ax = plt.subplots()
    for key, values in losses.items():
        plt.errorbar(range(len(values)), values, yerr=losses_std[key], label=key)

    plt.xlabel('# epochs')
    plt.ylabel('loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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



