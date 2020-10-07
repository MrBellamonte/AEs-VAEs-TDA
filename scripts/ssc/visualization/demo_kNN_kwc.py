from random import uniform

import torch

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles

from src.topology.witness_complex import WitnessComplex

BASE_PATH = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/kNN_kwc/'

from math import sin, cos, radians, pi, sqrt

def annulus(n, rmin, rmax,label = 0, seed = 0, c_x = 0, c_y = 0):
    np.random.seed(seed)
    phi = np.random.uniform(0, 2*np.pi, n)
    r = np.sqrt(np.random.uniform(rmin**2, rmax**2, n))
    x = r*np.cos(phi) + c_x
    y = r*np.sin(phi) + c_y
    label = np.ones((n))*label
    return pd.DataFrame({'x': x, 'y': y, 'label': label})

def make_df(X,y):
    return pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'label': y})

def make_scatter(df, title = None, show = True, name = None,base_path = BASE_PATH):
    sns.set_style("white")
    palette = [sns.color_palette("RdBu_r", 4)[0],sns.color_palette("RdBu_r", 4)[3]]
    custom_palette = sns.set_palette(palette)



    sns_plot = sns.scatterplot(data=df, x = df['x'], y = 'y', hue = 'label',
                     edgecolor="none", legend=False, palette = custom_palette)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.xticks([], " ")
    plt.yticks([], " ")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")

    if show:
        plt.show()

    if name != None:
        fig = sns_plot.get_figure()
        fig.savefig(base_path + name + '.pdf', dpi = 200)

    plt.close()

def plot_graph_kNN(df, k, df_witnesses = None, witnesses_dist = None, title = None, show = True, name = None, nc = 2,base_path = BASE_PATH):
    if witnesses_dist is not None:
        X_dist = witnesses_dist
        dist = True
    else:
        dist = False
        X_dist = df[['x', 'y']].to_numpy()



    kNN_ind_sparse = get_kNN_torch(X_dist, dist=dist)
    kNN = collect_kNN(kNN_ind_sparse, k)


    fig, ax = plt.subplots(1, 1)
    sns.set_style("white")
    if nc == 2:
        palette_1 = [sns.color_palette("RdBu_r", 4)[0], sns.color_palette("RdBu_r", 4)[3]]
    else:
        palette_1 = [sns.color_palette("RdBu_r", 4)[0]]

    #custom_palette = sns.set_palette(palette)

    if df_witnesses is not None:
        if nc == 2:
            palette_2 = [sns.color_palette("RdBu_r", 4)[1], sns.color_palette("RdBu_r", 4)[2]]
        else:
            palette_2 = [sns.color_palette("RdBu_r", 4)[1]]
        sns.scatterplot(data=df_witnesses, x='x', y='y', hue='label', ax=ax,
                        edgecolor="none", legend=False, palette=palette_2,zorder=0)
    sns.despine(left=True, bottom=True)
    plt.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.xticks([], " ")
    plt.yticks([], " ")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    for pairing in kNN:
        ax.plot([df['x'][pairing[0]], df['x'][pairing[1]]],
                [df['y'][pairing[0]], df['y'][pairing[1]]], color='grey',zorder=5)


    sns.scatterplot(data=df, x='x', y='y', hue='label',ax=ax,
                               edgecolor="none", legend=False, palette=palette_1, zorder=10)
    if show:
        plt.show()
    if name != None:
        fig.savefig(base_path + name+ '.pdf', dpi = 200)

    plt.close()


def get_kNN_torch(X,dist = False):
    if dist:
        X_dist = torch.from_numpy(X)
    else:
        X = torch.from_numpy(X)
        X_dist = torch.norm(X[:, None]-X, dim=2, p=2)
    sorted, indices = torch.sort(X_dist)

    #kNN_mask = torch.zeros((X.size(0), X.size(0),)).scatter(1, indices[:, 1:(k+1)],1)
    return np.array(indices)

def collect_kNN(indices, k):
    pairs = []
    for ind in indices:
        for j in ind[1:(k+1)]:
            pairs.append([ind[0],j])
    return pairs


if __name__ == "__main__":
    #### Double Annulus
    small = [0.5,0.8]
    large = [1, 1.3]

    # Manifold Annulus
    n_manifold = 512
    seed_manifold = 1
    df_an1 = annulus(n_manifold,large[0],large[1], seed = seed_manifold)
    df_an2 = annulus(n_manifold, small[0], small[1], label = 1,seed =  (seed_manifold + 1))
    df_an = df_an1.append(df_an2,ignore_index=True)
    make_scatter(df_an, name = 'double_annulus/annulus_manifold_512_s{s}'.format(s = seed_manifold))




    # radius = [1.65, 1.9]
    # radius_tight = [1.75, 1.85]
    # # Manifold Annulus
    # create_witness = True
    # c_y = 3.4
    #
    # x_min = -0.2
    # x_max = 0.2
    #
    # n_manifold = 256
    # seed_manifold = 1
    # df1 = annulus(n_manifold,radius[0],radius[1],seed = seed_manifold)
    # df2 = annulus(n_manifold,radius[0],radius[1],seed = seed_manifold, c_x = 0, c_y = c_y)
    # df = df1.append(df2,ignore_index=True)
    #
    #
    # df_selected = df[((df.x >= x_min) & (df.x<= x_max) & (df.y>= 0)& (df.y<= c_y))== False]
    # make_scatter(df_selected, name = 'almost8/almost8_manifold_1024_s{s}'.format(s = seed_manifold))
    #
    # # Approx Annulus
    # n_sparse = 16
    # seed_sparse = 10
    # for seed_sparse in [1,3,4,5,6,7,8,9]:
    #     df1_sparse = annulus(n_sparse, radius_tight[0], radius_tight[1], seed=(seed_sparse+23))
    #     df2_sparse = annulus(n_sparse, radius_tight[0], radius_tight[1], seed=seed_sparse, c_x=0, c_y=c_y)
    #     df_sparse = df1_sparse.append(df2_sparse, ignore_index=True)
    #
    #     df_selected_sparse = df_sparse[((df_sparse.x >= x_min) & (df_sparse.x <= x_max) & (df_sparse.y >= 0) & (df_sparse.y <= c_y)) == False]
    #     make_scatter(df_selected_sparse,
    #                  name='almost8/almost8_sparse_{n}_s{s}'.format(n=n_sparse,s=seed_sparse))
    #     df_selected_sparse = df_selected_sparse.reset_index()
    #     X_sparse = df_selected_sparse[['x', 'y']].to_numpy()
    #
    #     # Create kNN
    #     plot_graph_kNN(df_selected_sparse, k=1,
    #                    name='almost8/almost8_1-NN_sparse_{nsl}_s{s}'.format(
    #                        nsl=n_sparse, s=seed_sparse))
    #     plot_graph_kNN(df_selected_sparse, k=2,
    #                    name='almost8/almost8_2-NN_sparse_{nsl}_s{s}'.format(
    #                        nsl=n_sparse, s=seed_sparse))
    #     plot_graph_kNN(df_selected_sparse, k=3,
    #                    name='almost8/almost8_3-NN_sparse_{nsl}_s{s}'.format(
    #                        nsl=n_sparse, s=seed_sparse))
    #     plot_graph_kNN(df_selected_sparse, k=4,
    #                    name='almost8/almost8_4-NN_sparse_{nsl}_s{s}'.format(
    #                        nsl=n_sparse, s=seed_sparse))
    #
    #     # create k-WC
    #     if create_witness:
    #         n_witness = 256
    #         seed_witnesses = seed_sparse+42
    #
    #         df1_witnesses = annulus(n_witness, radius[0], radius[1], seed=(seed_witnesses+23))
    #         df2_witnesses = annulus(n_witness, radius[0], radius[1], seed=seed_witnesses, c_x=0, c_y=c_y)
    #         df_witnesses = df1_witnesses.append(df2_witnesses, ignore_index=True)
    #
    #         df_selected_witnesses = df_witnesses[((df_witnesses.x >= x_min) & (df_witnesses.x <= x_max) & (df_witnesses.y >= 0) & (df_witnesses.y <= c_y)) == False]
    #         df_selected_witnesses['label'] = df_selected_witnesses['label']-1
    #         make_scatter(df_selected_witnesses,
    #                      name='almost8/almost8_{n}_sw{sw}'.format(n=n_witness,sw=seed_witnesses))
    #
    #         X_witness = df_selected_witnesses[['x', 'y']].to_numpy()
    #
    #         wc_s = WitnessComplex(landmarks=X_sparse, witnesses=X_witness)
    #         wc_s.compute_simplicial_complex(1, create_metric=True)
    #
    #         plot_graph_kNN(df_selected_sparse, k=1, df_witnesses=df_selected_witnesses,
    #                        witnesses_dist=wc_s.landmarks_dist,
    #                        name='almost8/kWC1_sp{nl}_{nw}_s{s}_sw{sw}'.format(nl=n_sparse,
    #                                                                                    nw=n_witness,
    #                                                                                    s=seed_sparse,
    #                                                                                    sw=seed_witnesses))
    #         plot_graph_kNN(df_selected_sparse, k=2, df_witnesses=df_selected_witnesses,
    #                        witnesses_dist=wc_s.landmarks_dist,
    #                        name='almost8/kWC2_sp{nl}_{nw}_s{s}_sw{sw}'.format(nl=n_sparse,
    #                                                                                    nw=n_witness,
    #                                                                                    s=seed_sparse,
    #                                                                                    sw=seed_witnesses))
    #         plot_graph_kNN(df_selected_sparse, k=3, df_witnesses=df_selected_witnesses,
    #                        witnesses_dist=wc_s.landmarks_dist,
    #                        name='almost8/kWC3_sp{nl}_{nw}_s{s}_sw{sw}'.format(nl=n_sparse,
    #                                                                                    nw=n_witness,
    #                                                                                    s=seed_sparse,
    #                                                                                    sw=seed_witnesses))
    #         plot_graph_kNN(df_selected_sparse, k=4, df_witnesses=df_selected_witnesses,
    #                        witnesses_dist=wc_s.landmarks_dist,
    #                        name='almost8/kWC4_sp{nl}_{nw}_s{s}_sw{sw}'.format(nl=n_sparse,
    #                                                                                    nw=n_witness,
    #                                                                                    s=seed_sparse,
    #                                                                                    sw=seed_witnesses))

# Approx Annulus
    n_sparse_l = 82
    n_sparse_s = 48
    seed_sparse = 10
    for seed_sparse in [19]:

        df_an1_sparse = annulus(n_sparse_l,large[0],large[1], seed = seed_sparse)
        df_an2_sparse = annulus(n_sparse_s, small[0], small[1], label = 1,seed =  (seed_sparse + 1))
        df_an_sparse = df_an1_sparse.append(df_an2_sparse,ignore_index=True)
        make_scatter(df_an_sparse, name = 'double_annulus/annulus_sparse_64_36_s{s}'.format(s = seed_sparse))

        X_sparse = df_an_sparse[['x','y']].to_numpy()

        # Create kNN
        plot_graph_kNN(df_an_sparse, k=1, name='double_annulus/annulus_1-NN_sparse_{nsl}_{nss}_s{s}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse))
        plot_graph_kNN(df_an_sparse, k=2, name='double_annulus/annulus_2-NN_sparse_{nsl}_{nss}_s{s}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse))
        plot_graph_kNN(df_an_sparse, k=3, name='double_annulus/annulus_3-NN_sparse_{nsl}_{nss}_s{s}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse))
        plot_graph_kNN(df_an_sparse, k=4, name='double_annulus/annulus_4-NN_sparse_{nsl}_{nss}_s{s}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse))

        # create k-WC
        n_witness_l = 244
        n_witness_s = 140
        seed_witnesses = seed_sparse + 42

        df_an1_witness = annulus(n_witness_l, large[0], large[1], label=2 ,seed=seed_witnesses)
        df_an2_witness = annulus(n_witness_s, small[0], small[1], label=3, seed=(seed_witnesses+1))
        df_an_witness = df_an1_witness.append(df_an2_witness, ignore_index=True)
        df_an_witness_plot = df_an_witness
        df_an_witness_plot['label'] = df_an_witness_plot['label']-1
        make_scatter(df_an_witness_plot, name='double_annulus/witness_256_146_sw{sw}'.format(sw = seed_witnesses))

        X_witness = df_an_witness[['x', 'y']].to_numpy()

        wc_s = WitnessComplex(landmarks=X_sparse,witnesses=X_witness)
        wc_s.compute_simplicial_complex(1, create_metric = True)

        plot_graph_kNN(df_an_sparse, k=1, df_witnesses = df_an_witness, witnesses_dist=wc_s.landmarks_dist, name='double_annulus/kWC1_sp{nsl}_{nss}_s{s}_sw{sw}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse,sw = seed_witnesses))
        plot_graph_kNN(df_an_sparse, k=2, df_witnesses = df_an_witness, witnesses_dist=wc_s.landmarks_dist, name='double_annulus/kWC2_sparse_{nsl}_{nss}_s{s}_sw{sw}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse,sw = seed_witnesses))
        plot_graph_kNN(df_an_sparse, k=3, df_witnesses = df_an_witness, witnesses_dist=wc_s.landmarks_dist, name='double_annulus/kWC3_sparse_{nsl}_{nss}_s{s}_sw{sw}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse,sw = seed_witnesses))
        plot_graph_kNN(df_an_sparse, k=4, df_witnesses=df_an_witness,
                       witnesses_dist=wc_s.landmarks_dist, name='double_annulus/kWC4_sparse_{nsl}_{nss}_s{s}_sw{sw}'.format(nsl = n_sparse_l, nss = n_sparse_s,s = seed_sparse,sw = seed_witnesses))









