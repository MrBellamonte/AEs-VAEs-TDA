import datetime
import os

from src.models.COREL.train_engine import simulator_COREL, simulator_COREL_2

from scripts.ssc.COREL.config_library import test_grid_local

if __name__ == "__main__":
    # # create root_dir
    # root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator/bugfix/1'
    # now = datetime.datetime.now()
    # path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    # try:
    #     os.makedirs(path)
    # except:
    #     pass
    #
    #
    # simulator_COREL(config_test, path, verbose=True, data_constant=True)

    simulator_COREL_2(test_grid_local)
