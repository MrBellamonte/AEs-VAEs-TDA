import datetime
import os
import threading

from src.model.COREL.train_engine import simulator

from scripts.ssc.config_library import *


if __name__ == "__main__":

    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/test_simulator'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass

    threading.Thread(target=simulator(config_grid_testSpheres, path, verbose = True)).start()
    print('---')
    threading.Thread(target=simulator(config_grid_test_tshinge, path, verbose = True)).start()

    processes = (simulator(config_grid_test_tshinge, path, verbose = True),simulator(config_grid_test_tshinge, path, verbose = True))