import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/animations'
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

# need: function to plot VR-filtration form a scene a and a filtration value R


def init():
    line.set_data([], [])
    return line,
def animate(i):

    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)


anim.save(os.path.join(path_to_save, 'sine_wave.gif'), writer='imagemagick')