from scripts.ssc.TopoAE.config_libraries.spheres import spheres_euler_seed6_parallel_shuffled
from src.models.TopoAE.train_engine import simulator_TopoAE
from joblib import Parallel, delayed

if __name__ == "__main__":
    Parallel(n_jobs=9)(delayed(simulator_TopoAE)(config) for config in spheres_euler_seed6_parallel_shuffled)