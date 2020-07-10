import datetime
import os

from src.model.COREL.train_engine import simulator


from scripts.ssc.config_library import *


if __name__ == "__main__":





    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/tshinge'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass

    simulator(config_grid_Spheres_TwoSidedHinge, path, verbose=True)

    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/hinge'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    simulator(config_grid_Spheres_Hinge, path, verbose=True)

    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/l1'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass

    simulator(config_grid_Spheres_L1, path, verbose=True)

