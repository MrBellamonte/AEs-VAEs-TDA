from scripts.ssc.TopoAE.topoae_config_library import test_grid_local1_MBP, test_grid_local2_MBP
from src.models.TopoAE.train_engine import simulator_TopoAE
from joblib import Parallel, delayed

if __name__ == "__main__":

    Parallel(n_jobs=2)(delayed(simulator_TopoAE)(config) for config in [test_grid_local1_MBP,test_grid_local2_MBP])
