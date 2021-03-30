from scripts.ssc.models.TopoAE.config_libraries.local_configs.mnist2 import mnist_test_loc
from scripts.ssc.models.TopoAE.config_libraries.swissroll import swissroll_testing
from src.models.TopoAE.train_engine import simulator_TopoAE

if __name__ == "__main__":
    simulator_TopoAE(mnist_test_loc.configs_from_grid()[0])
