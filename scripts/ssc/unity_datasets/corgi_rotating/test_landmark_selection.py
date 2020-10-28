import random

import numpy as np

positions = np.array(random.sample(range(360),360))
deg_landmarks = np.linspace(0,359,30,dtype=int)


ind_l = np.in1d(positions, deg_landmarks).nonzero()[0]

print(np.array(sorted(positions[ind_l])))

