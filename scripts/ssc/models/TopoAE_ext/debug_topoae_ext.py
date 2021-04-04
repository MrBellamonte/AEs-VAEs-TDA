import itertools
import random

from scripts.ssc.models.TopoAE_ext.config_libraries.euler_configs.mnist import (
    mnist_s838_256_1,
    mnist_s838_512_1, mnist_s838_1024_1)
from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.mnist_ import mnist_test2, mnist_test256
from scripts.ssc.models.TopoAE_ext.config_libraries.swissroll import debug
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":
    config_list = list(itertools.chain(*[config_grid.configs_from_grid() for config_grid in
                                         [mnist_s838_256_1, mnist_s838_512_1,
                                          mnist_s838_1024_1]]))
    print(config_list)
    random.shuffle(config_list)
    #mnist_s838_1 = random.shuffle(config_list)
    print(config_list)

    # config = mnist_test2.configs_from_grid()[0]
    # simulator_TopoAE_ext(config)
