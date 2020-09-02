from scripts.ssc.TopoAE.config_libraries.swissroll import \
    (
    swissroll_midsize_lowbs_local_seed1_parallel_shuffled, swissroll_benchmark,
    swissroll_asymmetric_2, swissroll_asymmetric_push)
from scripts.ssc.TopoAE.topoae_config_library import (
    test_grid_local1_MBP, test_grid_local2_MBP,
    test_grid_local)
from src.models.TopoAE.train_engine import simulator_TopoAE
from joblib import Parallel, delayed

if __name__ == "__main__":

    Parallel(n_jobs=8)(delayed(simulator_TopoAE)(config) for config in swissroll_asymmetric_push)
