from scripts.ssc.TopoAE_ext.config_libraries.local_configs.mnist import mnist_test2, mnist_test256
from scripts.ssc.TopoAE_ext.config_libraries.swissroll import debug
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":

    config = mnist_test256.configs_from_grid()[0]
    simulator_TopoAE_ext(config)
