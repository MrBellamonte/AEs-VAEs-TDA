import datetime
import os


from scripts.ssc.TopoAE.topoae_config_library import moor_config_approx_1

from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":

    # create root_dir
    root_dir = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/Spheres/TopoAE/moor_approx_1'
    now = datetime.datetime.now()
    path = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(path)
    except:
        pass


    simulator_TopoAE(moor_config_approx_1,path,verbose= True, data_constant=True, num_threads=2)