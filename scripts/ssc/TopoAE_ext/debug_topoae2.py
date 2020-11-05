from scripts.ssc.TopoAE_ext.config_libraries.local_configs.mnist import mnist_test2, mnist_test3
from scripts.ssc.TopoAE_ext.config_libraries.swissroll import debug
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":

    configs = mnist_test3.configs_from_grid()
    for config in configs:
        simulator_TopoAE_ext(config)
