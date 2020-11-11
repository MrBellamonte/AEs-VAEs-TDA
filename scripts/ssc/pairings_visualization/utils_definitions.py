import numpy as np
import matplotlib.pyplot as plt

#PATH_ROOT_SWISSROLL = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/'
PATH_ROOT_SWISSROLL = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/'
#PATH_ROOT_SWISSROLL = '/Users/simons/polybox/Studium/20FS/MT/plots_/test'

def make_plot(data, pairings, color,name = 'noname', path_root = PATH_ROOT_SWISSROLL, knn = False, dpi = 200, show = False, angle = 5,cmap = plt.cm.Spectral):
    ax = plt.gca(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=100, cmap=cmap)


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
    #ax.axis('equal')
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    if path_root is not None:
        fig = ax.get_figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0, hspace = 0)
        fig.savefig(path_root+'btightplotsc_{}'.format(name)+'.pdf', dpi=dpi, bbox_inches='tight',
                    pad_inches=0)
        bbox = fig.bbox_inches.from_bounds(1, 1, 5, 5)
        fig.savefig(path_root + 'b5plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)
        bbox = fig.bbox_inches.from_bounds(1, 1, 4, 4)
        fig.savefig(path_root + 'b4plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

        bbox = fig.bbox_inches.from_bounds(1, 1, 3, 3)
        fig.savefig(path_root + 'b3plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

        bbox = fig.bbox_inches.from_bounds(1, 1, 6, 6)
        fig.savefig(path_root + 'b6plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

    if show:
        plt.show()
    plt.close()