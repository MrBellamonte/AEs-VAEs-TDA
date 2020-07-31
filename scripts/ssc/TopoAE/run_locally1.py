import datetime
import os


from scripts.ssc.TopoAE.topoae_config_library import *

from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":

    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/TopoAE/Spheres'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    simulator_TopoAE(moor_config_approx,path,verbose= True, data_constant=True, num_threads=2)