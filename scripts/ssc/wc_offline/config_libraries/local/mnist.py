from scripts.ssc.wc_offline.config_libraries.global_register_definitions import (
    PATH_GR_MNIST_EULER,
    PATH_GR_MNIST_LOCAL)
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import MNIST

mnist_1 = ConfigWC_Grid(
    dataset = [MNIST()],
    sampling_kwargs = [dict()],
    batch_size=[128],
    wc_kwargs=[dict()],
    eval_size=[0.14], # not a lot of data "wasted for bs = 4096 & could go to bs=8192
    n_jobs = [8],
    seed = [1],
    global_register = PATH_GR_MNIST_LOCAL,
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/mnist',
    verbose = True
)
