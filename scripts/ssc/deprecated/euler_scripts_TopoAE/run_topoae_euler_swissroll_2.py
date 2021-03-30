from scripts.ssc.models.TopoAE import swissroll_midsize_euler1
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(swissroll_midsize_euler1)
