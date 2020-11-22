from scripts.ssc.TopoAE_ext.config_libraries.colab_configs.mnist import mnist_test512_cuda
from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.unity_posttrain import \
    (
    rotopenai_1_local2, rotopenai_2_local)
from scripts.ssc.TopoAE_ext.config_libraries.local_configs.mnist import (
    mnist_test2, mnist_test3,
    mnist_test256_1024_leonhard, mnist_test256, mnist_test)
from scripts.ssc.TopoAE_ext.config_libraries.local_configs.unity import (
    rotopenai_test,
    xytrans_openai_test)
from scripts.ssc.TopoAE_ext.config_libraries.swissroll import debug
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":
    configs = mnist_test.configs_from_grid()

    for config in configs:
        simulator_TopoAE_ext(config)
