from scripts.ssc.TopoAE.topoae_config_library import *
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    path = '/cluster/home/schsimo/MT/output/280720'
    simulator_TopoAE(eulergrid_280720_2,path,verbose= True, data_constant=True)
