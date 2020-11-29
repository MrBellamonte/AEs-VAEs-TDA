from scripts.ssc.models.TopoAE_ext.config_libraries.colab_configs.mnist import mnist_test512_cuda
from scripts.ssc.models.TopoAE_ext.config_libraries.euler_configs.unity_posttrain import rotopenai_1_local
from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.mnist import (
    mnist_test2, mnist_test3,
    mnist_test256_1024_leonhard, mnist_test256, mnist_test_3d)
from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.unity import rotopenai_test
from scripts.ssc.models.TopoAE_ext.config_libraries.swissroll import debug, swissroll_testing
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":
    configs = swissroll_testing.configs_from_grid()

    for config in configs:
        simulator_TopoAE_ext(config)
