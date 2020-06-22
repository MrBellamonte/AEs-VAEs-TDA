import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



def plot_simplicial_complex_2D(simp_complex: list, points: np.ndarray, scale: float):
    #todo: allow custom style?


    # 2-simplicies
    triangles = np.array(
        [s[0] for s in simp_complex if len(s[0]) == 3 and s[1] <= scale])
    triangle_patches = []
    for triangle in triangles:
        coord = np.column_stack(
            (points[triangle, 0].reshape(3, ), points[triangle, 1].reshape(3, )))
        polygon = Polygon(coord, True)
        triangle_patches.append(polygon)

    p = PatchCollection(triangle_patches, cmap=matplotlib.cm.jet, alpha=0.4)
    colors = 50*np.ones(len(triangle_patches))
    p.set_array(np.array(colors))

    plt.gca().add_collection(p)

    # 1-simplicies
    coord_1s = np.array(
        [s[0] for s in simp_complex if len(s[0]) == 2 and s[1] <= scale])
    for idxs in coord_1s:
        plt.plot(points[idxs, 0], points[idxs, 1], color='darkblue')

    # 0-simplicies
    plt.scatter(points[:, 0], points[:, 1], color='indigo', zorder=10)

    plt.show()

