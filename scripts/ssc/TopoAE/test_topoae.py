from scripts.ssc.TopoAE.config_libraries.swissroll import \
    (
    swissroll_midsize_lowbs_euler_seed1_parallel, swissroll_midsize_midbs_euler_seed1_1)
from scripts.ssc.TopoAE.topoae_config_library import test_grid_local
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(test_grid_local)

    #simulator_TopoAE_parallel(test_grid_local, n_jobs=2)



