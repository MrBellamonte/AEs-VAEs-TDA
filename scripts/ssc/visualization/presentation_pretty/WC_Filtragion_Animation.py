import math
import os
from typing import List


from celluloid import Camera
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import numpy as np
from scripts.ssc.visualization.demo_kNN_kwc import annulus
from src.topology.witness_complex import WitnessComplex
from src.utils.plots import plot_simplicial_complex_2D


def wc_plot(simplex_tree: list, landmarks: np.ndarray, witnesses: np.ndarray,witnesses_only, filtration_value: float ,path_to_save = None, name = None):
    fig, ax = plt.subplots()

    # labels
    label_w = 'witness'
    label_l = 'landmark'
    label_s0 = '0-simplex'
    label_s1 = '1-simplex'
    label_s2 = '2-simplex'



    # color config
    palette = sns.color_palette("muted")
    color_landmarks = palette[-1]
    color_landmarks = palette[4]
    color_witnesses = palette[3]


    # palette_s = sns.cubehelix_palette(start=.5, rot=-.5)
    # palette_s = sns.dark_palette("blue", reverse = True)
    #palette_s = sns.color_palette("crest")
    #palette_s = sns.color_palette("Greens")
    palette_s = sns.color_palette("Blues")
    color_s0 = palette_s[-1]
    color_s1 = palette_s[-3]
    color_s2 = palette_s[0]
    transparancy = 0.75


    #
    #plt.scatter(landmarks[:,0], landmarks[:,1], color = color_landmarks,zorder=9, label = label_l)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color=color_s0, zorder=10,
                label=label_l)

    #plot witnesses
    plt.scatter(witnesses_only[:, 0], witnesses_only[:, 1], color = color_witnesses,zorder=10,label = label_w)

    #plot circle around witnesses
    for i in range(X_witnesses.shape[0]):
        a_circle = plt.Circle((witnesses[i,0], witnesses[i,1], 0), filtration_value, color = 'grey',fill = False,zorder=8)
        ax.add_artist(a_circle)


    #plot simplicies
    simplex0 = []
    simplex1 = []
    simplex2 = []
    all_s = []
    for element in simplex_tree.get_skeleton(3):
        all_s.append(element)
        if element[1]<= filtration_value:
            if len(element[0]) == 1:
                simplex0.append(element[0])
            elif len(element[0]) == 2:
                simplex1.append(element[0])
            elif len(element[0]) == 3:
                simplex2.append(element[0])
            else:
                pass

    # # plot 0-simplices, i.e. landmarks in dark blue
    # if len(simplex0) >0:
    #     plt.scatter(landmarks[simplex0,0], landmarks[simplex0,1], color = color_s0,zorder=10,label = label_s0)


    # plot 1-simplice
    i = 0
    for simplex in simplex1:
        if i == 0:
            plt.plot(landmarks[simplex, 0], landmarks[simplex, 1], color=color_s1,zorder=0, label = label_s1)
        else:
            plt.plot(landmarks[simplex, 0], landmarks[simplex, 1], color=color_s1, zorder=0)
        i += 1
    # plot 2-simplex
    triangle_patches = []
    for idx in simplex2:
        coord = np.column_stack(
            (landmarks[idx, 0].reshape(3, ), landmarks[idx, 1].reshape(3, )))
        polygon = Polygon(coord, True)
        triangle_patches.append(polygon)

    cm = LinearSegmentedColormap.from_list(name = 'mylist',colors = [color_s2,color_s2], N=1)
    p = PatchCollection(triangle_patches, cmap=cm, alpha = transparancy, zorder = 0)
    colors = 1000000*np.ones(len(triangle_patches))
    p.set_array(np.array(colors))

    plt.gca().add_collection(p)



    #aesthetics
    sns.despine(left = True, bottom=True)
    plt.xticks([], " ")
    plt.yticks([], " ")
    plt.xlabel("")
    plt.ylabel("")
    margin = 3
    x_min = -0.25 - margin
    x_max = 4 + margin
    y_min = 0 - margin
    y_max = 4.25 + margin
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    handles, labels = ax.get_legend_handles_labels()
    if len(simplex2)>0:
        dummy_s2 = mpatches.Patch(color=palette_s[0], label=label_s2, alpha = transparancy)
        handles.append(dummy_s2)
        labels.append(label_s2)

    if len(handles) == 5:
        order = [2,1,3,0,4]

    elif len(handles) == 4:
        #order = [2, 1, 3, 0]
        order = [1, 2, 0, 3]
    elif len(handles) == 3:
        #order = [1, 0, 2]
        order = [1, 2, 0]
    elif len(handles) == 2:
        #order = [1, 0]
        order = [0, 1]
    else:
        pass




    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    # plot the legend

    plt.show()

    if path_to_save is not None:
        plot_name = name + '.pdf'
        path_to_save = os.path.join(path_to_save, plot_name)
        fig.savefig(path_to_save, dpi = 200)

def wc_plot_anmi(simplex_tree: list, landmarks: np.ndarray, witnesses: np.ndarray,witnesses_only, filtration_value: float):
    # labels
    label_w = 'witness'
    label_l = 'landmark'
    label_s1 = '1-simplex'
    label_s2 = '2-simplex'



    # color config
    palette = sns.color_palette("muted")
    color_witnesses = palette[3]

    palette_s = sns.color_palette("Blues")
    color_s0 = palette_s[-1]
    color_s1 = palette_s[-3]
    color_s2 = palette_s[0]
    transparancy = 0.75

    plt.scatter(landmarks[:, 0], landmarks[:, 1], color=color_s0, zorder=10,
                label=label_l)
    plt.scatter(witnesses_only[:, 0], witnesses_only[:, 1], color = color_witnesses,zorder=10,label = label_w)

    #plot circle around witnesses
    for i in range(X_witnesses.shape[0]):
        a_circle = plt.Circle((witnesses[i,0], witnesses[i,1], 0), filtration_value, color = 'grey',fill = False,zorder=8)
        ax.add_artist(a_circle)

    #plot simplicies
    simplex0 = []
    simplex1 = []
    simplex2 = []
    all_s = []
    for element in simplex_tree.get_skeleton(3):
        all_s.append(element)
        if element[1]<= filtration_value:
            if len(element[0]) == 1:
                simplex0.append(element[0])
            elif len(element[0]) == 2:
                simplex1.append(element[0])
            elif len(element[0]) == 3:
                simplex2.append(element[0])
            else:
                pass

    # plot 1-simplice
    i = 0
    for simplex in simplex1:
        if i == 0:
            plt.plot(landmarks[simplex, 0], landmarks[simplex, 1], color=color_s1,zorder=0, label = label_s1)
        else:
            plt.plot(landmarks[simplex, 0], landmarks[simplex, 1], color=color_s1, zorder=0)
        i += 1
    # plot 2-simplex
    triangle_patches = []
    for idx in simplex2:
        coord = np.column_stack(
            (landmarks[idx, 0].reshape(3, ), landmarks[idx, 1].reshape(3, )))
        polygon = Polygon(coord, True)
        triangle_patches.append(polygon)

    cm = LinearSegmentedColormap.from_list(name = 'mylist',colors = [color_s2,color_s2], N=1)
    p = PatchCollection(triangle_patches, cmap=cm, alpha = transparancy, zorder = 0)
    colors = 1000000*np.ones(len(triangle_patches))
    p.set_array(np.array(colors))

    plt.gca().add_collection(p)

    #aesthetics
    sns.despine(left = True, bottom=True)
    plt.xticks([], " ")
    plt.yticks([], " ")
    plt.xlabel("")
    plt.ylabel("")
    margin = 3
    x_min = -0.25 - margin
    x_max = 4 + margin
    y_min = 0 - margin
    y_max = 4.25 + margin
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    handles, labels = ax.get_legend_handles_labels()
    if len(simplex2)>0:
        dummy_s2 = mpatches.Patch(color=palette_s[0], label=label_s2, alpha = transparancy)
        handles.append(dummy_s2)
        labels.append(label_s2)
    #
    # if len(handles) == 5:
    #     order = [2,1,3,0,4]
    # elif len(handles) == 4:
    #     #order = [2, 1, 3, 0]
    #     order = [1, 2, 0, 3]
    # elif len(handles) == 3:
    #     #order = [1, 0, 2]
    #     order = [1, 2, 0]
    # elif len(handles) == 2:
    #     #order = [1, 0]
    #     order = [0, 1]
    # elif len(handles) == 6:
    #     order = [2,1,3,0,4,5]
    # else:
    #     pass
    #
    #
    #
    #
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    plt.legend( loc='upper right')

if __name__ == "__main__":
    X_landmarks = np.array([[-0.25, 1.5], [1.1, 0], [4, 1.4], [1.5, 3.5]])
    X_witnesses_ = np.array([[2.75, 3], [2.8, 0.55], [1, 1.5]])

    # X_landmarks = np.array([[0,0],[0.7,0.5],[1,1],[0.4,0.8]])*5
    # X_witnesses_ = np.array([[0.4,0.25],[0.9,0.75],[0.75,0.9],[0.25,0.4]])*5

    X_witnesses = np.vstack((X_witnesses_, X_landmarks))

    # X_tot = np.row_stack((X_landmarks, X_witnesses))
    np.random.seed(seed=3)
    sigma = 0.02
    rand_w = sigma*np.random.randn(X_witnesses.shape[0], X_witnesses.shape[1])
    rand_l = sigma*np.random.randn(X_landmarks.shape[0], X_landmarks.shape[1])

    X_landmarks = X_landmarks+rand_l
    X_witnesses = X_witnesses+rand_w

    witness_complex = WitnessComplex(X_landmarks, X_witnesses)
    witness_complex.compute_simplicial_complex(4, r_max=100, create_simplex_tree=True,
                                               create_metric=True)

    wc_plot(witness_complex.simplex_tree, X_landmarks, X_witnesses, filtration_value=1,
                  path_to_save=None,witnesses_only=X_witnesses_)

    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/animations'

    # need: function to plot VR-filtration form a scene a and a filtration value R
    fig, ax = plt.subplots()  # put outside
    #aesthetics


    filtration_values = []
    for element in witness_complex.simplex_tree.get_skeleton(2):
        filtration_values.append(element[1])


    camera = Camera(fig)
    for i in np.arange(0.0, max(filtration_values), 0.05):
        wc_plot_anmi(witness_complex.simplex_tree, X_landmarks, X_witnesses, filtration_value=i,witnesses_only=X_witnesses_)
        camera.snap()
    animation = camera.animate()
    animation.save(os.path.join(path_to_save, 'wc_anim.gif'), writer='imagemagick')

    #
    # anim = FuncAnimation(fig, animate, init_func=init,
    #                      frames=2, interval=0.1, blit=True)
    #
    # anim.save(os.path.join(path_to_save, 'wc_anim.gif'), writer='imagemagick')