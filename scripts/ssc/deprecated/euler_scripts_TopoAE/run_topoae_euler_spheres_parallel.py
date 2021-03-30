from scripts.ssc.models.TopoAE import spheres_lowmemory_midbs_euler_seed2_1, spheres_lowmemory_midbs_euler_seed2_2,spheres_lowmemory_midbs_euler_seed2_3, spheres_lowmemory_midbs_euler_seed2_4
from src.models.TopoAE.train_engine import simulator_TopoAE
from joblib import Parallel, delayed

if __name__ == "__main__":
    Parallel(n_jobs=4)(delayed(simulator_TopoAE)(config) for config in [spheres_lowmemory_midbs_euler_seed2_1,spheres_lowmemory_midbs_euler_seed2_2,spheres_lowmemory_midbs_euler_seed2_3,spheres_lowmemory_midbs_euler_seed2_4])