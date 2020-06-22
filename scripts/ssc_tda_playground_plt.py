import cechmate
import matplotlib.pyplot as plt
import matplotlib
import tadasets
import gudhi
import numpy as np
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


from utils.plots import plot_simplicial_complex_2D

data = tadasets.infty_sign(n=40, noise=0.07)

r = cechmate.Alpha()
simp = r.build(X = data)

plot_simplicial_complex_2D(simp, data, 0.2)