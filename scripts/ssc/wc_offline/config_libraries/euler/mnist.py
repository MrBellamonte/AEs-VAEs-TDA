from scripts.ssc.wc_offline.config_libraries.global_register_definitions import PATH_GR_MNIST_EULER
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll, MNIST

mnist_1 = ConfigWC_Grid(
    dataset = [MNIST()],
    sampling_kwargs = [dict(n_samples = 2560)],
    batch_size=[64,128,256,512,1024,2048,4096],
    wc_kwargs=[dict()],
    eval_size=[0.14], # not a lot of data "wasted for bs = 4096 & could go to bs=8192
    n_jobs = [10],
    seed = [838],
    global_register = PATH_GR_MNIST_EULER,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist',
    verbose = True
)
