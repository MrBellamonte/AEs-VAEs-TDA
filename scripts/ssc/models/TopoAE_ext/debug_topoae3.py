from scripts.ssc.models.TopoAE_ext.config_libraries.colab_configs.mnist import mnist_test512_cuda
from scripts.ssc.models.TopoAE_ext.config_libraries.euler_configs.unity_posttrain import \
    (
    rotopenai_1_local2, rotopenai_2_local)
from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.mnist_ import (
    mnist_test3,
    mnist_test256, mnist_test)
from scripts.ssc.models.TopoAE_ext.config_libraries.swissroll import debug
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":
    configs = mnist_test3.configs_from_grid()


    simulator_TopoAE_ext(configs[0])
