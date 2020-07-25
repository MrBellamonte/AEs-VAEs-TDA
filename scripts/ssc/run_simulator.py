import datetime
import os

from src.models.COREL.train_engine import simulator_COREL


from scripts.ssc.config_library import *


if __name__ == "__main__":





    # root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_n3_250/tshinge'
    # now = datetime.datetime.now()
    # path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    # try:
    #     os.makedirs(path)
    # except:
    #     pass
    #
    # simulator_COREL(config_grid_Spheres_n3_250_tshinge, path, verbose=True, data_constant = True)
    # #
    # # root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/hinge'
    # # now = datetime.datetime.now()
    # # path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    # # try:
    # #     os.makedirs(path)
    # # except:
    # #     pass
    #
    #
    # #simulator(config_grid_Spheres_Hinge, path, verbose=True)
    #
    # root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_n3_250/l1'
    # now = datetime.datetime.now()
    # path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    # try:
    #     os.makedirs(path)
    # except:
    #     passpi
    #
    # simulator_COREL(config_grid_Spheres_n3_250_l1, path, verbose=True, data_constant = True)

    path = 'PycharmProjects/MT-VAEs-TDA/output_simulator/spheres_default/l1/2020-07-17'

    simulator_COREL(config_grid_Spheres_benchmark,path,verbose= True, data_constant=True)
