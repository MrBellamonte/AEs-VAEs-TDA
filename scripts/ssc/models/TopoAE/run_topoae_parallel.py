from scripts.ssc.models.TopoAE import \
    (
    swissroll_midsize_lowbs_local_seed1_parallel_shuffled, swissroll_testing_parallel,
    swissroll_multiseed_parallel)
from src.models.TopoAE.train_engine import simulator_TopoAE
from joblib import Parallel, delayed

if __name__ == "__main__":

    Parallel(n_jobs=48)(delayed(simulator_TopoAE)(config) for config in swissroll_multiseed_parallel)
