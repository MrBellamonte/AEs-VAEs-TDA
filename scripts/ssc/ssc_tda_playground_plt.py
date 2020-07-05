import gudhi as gd
import tadasets

data = tadasets.infty_sign(n=35, noise = 0.07)

alpha_complex = gd.AlphaComplex(points = data)
alpha_simplex_tree = alpha_complex.create_simplex_tree()


print('run!')

# diag_alpha = alpha_simplex_tree.persistence()
# gd.plot_persistence_diagram(diag_alpha)
# plt.show()
# gd.plot_persistence_barcode(diag_alpha)
# plt.show()
#
#
#
# plot_simplicial_complex_2D(alpha_simplex_tree.get_skeleton(2), data, 0.18)

