import matplotlib.pyplot as plt
import tadasets
import gudhi
import numpy as np
import matplotlib.tri as mtri


# sample data
data = tadasets.infty_sign(n=40, noise=0.07)

# plot data (still plt)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[:, 0], data[:, 1])
fig.show()


# construct rips complex
rips_complex = gudhi.AlphaComplex(points=data)
simplex_tree = rips_complex.create_simplex_tree()


# get data ready for plotting

scale = 0.05
points = np.array(
   [rips_complex.get_point(i) for i in range(simplex_tree.num_vertices())])

# 1-simplicies
coord_1s = np.array(
    [s[0] for s in simplex_tree.get_skeleton(2) if len(s[0]) == 2 and s[1] <= scale])
for idxs in coord_1s:
    plt.plot(points[idxs, 0],points[idxs, 1])


# 2-simplicies
triangles = np.array(
    [s[0] for s in simplex_tree.get_skeleton(2) if len(s[0]) == 3 and s[1] <= scale])
triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles=triangles)

plt.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=np.ones(len(triangles)), edgecolors='k')

plt.triplot(triang, marker="o")

plt.show()
