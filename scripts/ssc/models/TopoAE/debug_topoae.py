from scripts.ssc.models.TopoAE.config_libraries.swissroll import swissroll_testing
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(swissroll_testing)