from scripts.ssc.TopoAE.topoae_config_library import swissroll_midsize_lowbs_euler_seed1_1
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(swissroll_midsize_lowbs_euler_seed1_1)
