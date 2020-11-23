from scripts.ssc.models.TopoAE import test_grid_local
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(test_grid_local)

    #simulator_TopoAE_parallel(test_grid_local, n_jobs=2)



