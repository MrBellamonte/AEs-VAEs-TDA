from scripts.ssc.models.TopoAE import swissroll_testing
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(swissroll_testing)

    #simulator_TopoAE_parallel(test_grid_local, n_jobs=2)



