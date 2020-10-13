import itertools
import numpy as np

from scripts.ssc.wc_offline.config_libraries.global_register_definitions import \
    PATH_GR_MNIST_EULER

seeds = np.repeat(838,5)
bs = [64,128,256,512,1024]

MNIST_838 = [
dict(uid = 'MNIST_offline-bs64-seed838-noiseNone-20738678', path_global_register = PATH_GR_MNIST_EULER),
dict(uid = 'MNIST_offline-bs128-seed838-noiseNone-4f608157', path_global_register = PATH_GR_MNIST_EULER),
dict(uid = 'MNIST_offline-bs256-seed838-noiseNone-4a5487de', path_global_register = PATH_GR_MNIST_EULER),
dict(uid = 'MNIST_offline-bs512-seed838-noiseNone-ced06774', path_global_register = PATH_GR_MNIST_EULER),
dict(uid = 'MNIST_offline-bs1024-seed838-noiseNone-6f31dea2', path_global_register = PATH_GR_MNIST_EULER)]