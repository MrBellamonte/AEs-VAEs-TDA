import cechmate
import gudhi
import tadasets
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plots import plot_simplicial_complex_2D

data = tadasets.infty_sign(n=60, noise=0.07)

# not really needed, since data is sampled randomly already
np.random.shuffle(data)

landmarks = data[:15]
witnesses = data[15:]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(witnesses[:, 0], witnesses[:, 1],c='black')
ax.scatter(landmarks[:, 0], landmarks[:, 1],c='red')
fig.show()


witness_complex = gudhi.EuclideanWitnessComplex(witnesses = witnesses, landmarks = landmarks)
simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=2, limit_dimension=3)


points = np.array(
   [witness_complex.get_point(i) for i in range(simplex_tree.num_vertices())])

print(simplex_tree.get_skeleton(2))
r = cechmate.Alpha()
simp = r.build(X = data)
#
plot_simplicial_complex_2D(simplex_tree.get_skeleton(2), data, 0.2)
plt.show()
simp_w = []
for s in simplex_tree.get_skeleton(2):
   simp_w.append(s)

edges = np.array(
        [s[0] for s in simplex_tree.get_skeleton(2) if len(s[0]) == 2])

print(simp_w)
print('---')
print(simp)


dgms = cechmate.phat_diagrams(simp_w, show_inf = True)
plot_diagrams(dgms)
plt.show()
#
#
# plt.subplot(121)
#
# plt.axis('square')
# plt.title("Point Cloud")
# plt.subplot(122)
# plot_diagrams(dgmsrips)
# plt.title("Rips Persistence Diagrams")
# plt.tight_layout()
# plt.show()

