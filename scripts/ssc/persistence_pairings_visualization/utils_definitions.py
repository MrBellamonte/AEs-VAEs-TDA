import matplotlib.pyplot as plt

#PATH_ROOT_SWISSROLL = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/'
PATH_ROOT_SWISSROLL = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/SwissRoll_pairings/'
#PATH_ROOT_SWISSROLL = '/Users/simons/polybox/Studium/20FS/MT/plots_/test'

def make_plot(data, pairings, color,name, path_root = PATH_ROOT_SWISSROLL):
    ax = plt.gca(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=100, cmap=plt.cm.Spectral)

    for pairing in pairings:
        ax.plot([data[pairing[0], 0], data[pairing[1], 0]],
                [data[pairing[0], 1], data[pairing[1], 1]],
                [data[pairing[0], 2], data[pairing[1], 2]], color='grey')



    ax.view_init(10, 90)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.show()
    fig = ax.get_figure()
    fig.savefig(path_root + 'plotsc_{}'.format(name) + '.pdf', dpi=200,bbox_inches = 'tight',
    pad_inches = 0)
    plt.close()