import os
from distutils.dir_util import copy_tree

dir_corgi = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/testing_unity/Unity_Rotblock-seed1-ConvAElarge_Unity-default-lr1_100-bs120-nep1-rlw1-tlw1024-mepush_active1-k1-rmax10-seed1-0c949dd0'
dir_dest = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/SwissRoll_precomputed'

copy_tree(dir_corgi, os.path.join(dir_dest,'corgi'))
