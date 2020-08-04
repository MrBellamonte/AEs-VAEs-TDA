from scripts.ssc.TopoAE.topoae_config_library import spheres_lowmemory_euler1
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(spheres_lowmemory_euler1)
