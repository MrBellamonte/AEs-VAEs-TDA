import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



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



