from scripts.ssc.TopoAE.topoae_config_library import test_grid
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    path = '/cluster/home/schsimo/MT/output/'
    simulator_TopoAE(test_grid,path,verbose= True, data_constant=True)
