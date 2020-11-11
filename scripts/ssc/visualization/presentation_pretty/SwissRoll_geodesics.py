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

from scripts.ssc.pairings_visualization.utils_definitions import make_plot
from src.datasets.datasets import SwissRoll



def plot_2Dscatter_wgeo(data, labels, path_to_save= None, title = None, show = False, geodesics = None):
    fig, ax = plt.subplots()
    sns_plot = sns.scatterplot(x = data[:, 0], y=data[:, 1], hue=labels, palette=plt.cm.viridis, marker=".",
                    s=30, edgecolor="none", legend=False, ax=ax)

    if geodesics is not None:
        sns.scatterplot(geodesics['x'],geodesics['y'],color='red',ax=ax,s=100)
        for i in [0,1]:
            ax.annotate(geodesics['names'][i], xy=(geodesics['x'][i], geodesics['y'][i]), xytext = geodesics['position'][i],weight="bold",color='red',fontsize=18)

        sns.lineplot(geodesics['x'],geodesics['y'],markers=True,color='red',ax=ax)

    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.title(title)

    if show:
        plt.show()

    if path_to_save != None:
        fig = sns_plot.get_figure()
        fig.savefig(path_to_save)

    plt.close()



def make_plot3D(data, pairings, color,name = 'noname', path_root = None, knn = False, dpi = 200, show = False, angle = 5,cmap = plt.cm.viridis, geodesics = None):
    ax = plt.gca(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=50, cmap=cmap)

    if geodesics is not None:
        ax.scatter(geodesics['x'],geodesics['y'],geodesics['z'], color='red', s=100)
        ax.plot(geodesics['x'], geodesics['y'], geodesics['z'], color='red')


    i = 0
    if pairings is None:
        pass
    else:
        for pairing in pairings:
            if knn:
                for ind in pairing:
                    ax.plot([data[i, 0], data[ind, 0]],
                            [data[i, 1], data[ind, 1]],
                            [data[i, 2], data[ind, 2]], color='grey')
            else:
                ax.plot([data[pairing[0], 0], data[pairing[1], 0]],
                        [data[pairing[0], 1], data[pairing[1], 1]],
                        [data[pairing[0], 2], data[pairing[1], 2]], color='grey')

            i += 1



    ax.view_init(angle, 90)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.margins(0, 0,0)

    #plt.axis('scaled')

    #find axis range

    axis_min = [min(data[:, i]) for i in [0,1,2]]
    axis_max = [max(data[:, i]) for i in [0, 1, 2]]
    margin = [(axis_max[i] - axis_min[i])*0.05 for i in [0, 1, 2]]

    axis_range = [np.array([axis_max[i]-margin[i], axis_max[i]+ margin[i]])for i in [0, 1, 2]]

    ax.set_xlim(np.array([axis_min[0]-margin[0], axis_max[0]+ margin[0]]))
    ax.set_ylim(np.array([axis_min[1]-margin[1], axis_max[1]+ margin[1]]))
    ax.set_zlim(np.array([axis_min[2]-margin[2], axis_max[2]+ margin[2]]))

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    if path_root is not None:
        fig = ax.get_figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0, hspace = 0)
        fig.savefig(os.path.join(path_root,'btightplotsc_{}_a{}'.format(name,angle)+'.pdf'), dpi=dpi, bbox_inches='tight',
                    pad_inches=0)
        # bbox = fig.bbox_inches.from_bounds(1, 1, 5, 5)
        # fig.savefig(path_root + 'b5plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        # pad_inches = 0)
        # bbox = fig.bbox_inches.from_bounds(1, 1, 4, 4)
        # fig.savefig(path_root + 'b4plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        # pad_inches = 0)
        #
        # bbox = fig.bbox_inches.from_bounds(1, 1, 3, 3)
        # fig.savefig(path_root + 'b3plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        # pad_inches = 0)
        #
        # bbox = fig.bbox_inches.from_bounds(1, 1, 6, 6)
        # fig.savefig(path_root + 'b6plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        # pad_inches = 0)

    if show:
        plt.show()
    plt.close()



if __name__ == "__main__":
    dataset = SwissRoll()

    Z_m,X,Y = dataset.sample_manifold(n_samples=2048)

    path_to_save = '/Users/simons/polybox/Studium/20FS/MT/Presentations/visualizations/SwissRoll'

    geodesics = dict(
        x = [7.5,14],#[4.8,11],#[7.5,14],#[6,13],
        y = [11,18],#20.5,18],#11,18],
        position = [(7.5, 9),(14, 19)],#[(5.5, 9.5),(13, 19)],
        names = ['P1','P2']
    )
    geodesics_3D = dict(
        x=geodesics['x']*np.cos(geodesics['x']),
        y=geodesics['y'],
        z=geodesics['x']*np.sin(geodesics['x']),
    )


    plot_2Dscatter_wgeo(Z_m, Y, show=True, geodesics=None,
                         path_to_save=os.path.join(path_to_save, 'manifold_many.pdf'))
    # plot_2Dscatter_wgeo(Z_m,Y,show = True,geodesics = geodesics, path_to_save=os.path.join(path_to_save,'manifold_wgeodesics.pdf'))
    #
    # make_plot3D(X,pairings=None,color=Y,path_root=path_to_save,angle=7,show=True, geodesics=geodesics_3D,name='3d_wgeodecis')
    # make_plot3D(X, pairings=None, color=Y, path_root=path_to_save, angle=7, show=True,
    #             geodesics=None, name='3d_wog')