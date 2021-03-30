from scripts.ssc.models.TopoAE import test_grid_euler
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(test_grid_euler)
