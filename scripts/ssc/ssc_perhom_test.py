import gudhi as gd
import matplotlib.pyplot as plt
from src.datasets.datasets import double_tours

data_double, labels = double_tours()

rips_complex_double = gd.RipsComplex(points = data_double,max_edge_length=2)
rips_simplex_tree_double = rips_complex_double.create_simplex_tree(max_dimension=4)


persistence_double = rips_simplex_tree_double.persistence()
persistence_double_filtered = [x for x in persistence_double if x[0]>1]


gd.plot_persistence_diagram(persistence_double_filtered)
plt.show()
gd.plot_persistence_barcode(persistence_double_filtered)
plt.show()

from src.datasets.shapes import torus
data_single, label = torus(n=500, c=6, a=4, label=0)
rips_complex_single = gd.RipsComplex(points = data_single,max_edge_length=2)
rips_simplex_tree_single = rips_complex_single.create_simplex_tree(max_dimension=4)

persistence_single = rips_simplex_tree_single.persistence()
print(persistence_single)
persistence_single_filtered = [x for x in persistence_single if x[0]>1]
print(persistence_single_filtered)
gd.plot_persistence_diagram(persistence_single_filtered)
plt.show()
gd.plot_persistence_barcode(persistence_single_filtered)
plt.show()
