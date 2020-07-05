import numpy as np

def label_gen_torus(theta, phi):
    '''
    Function to divide torus into 8 clusters
    '''
    if theta < np.pi:
        l1 = 0
    else:
        l1 = 4

    l2 = int(phi/(np.pi/2))

    return l1 + l2