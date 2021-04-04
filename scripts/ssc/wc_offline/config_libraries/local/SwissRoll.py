from scripts.ssc.wc_offline.config_libraries.global_register_definitions import PATH_GR_MNIST_LOCAL
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll

mnist_2 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict()],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.14], # not a lot of data "wasted for bs = 4096 & could go to bs=8192
    n_jobs = [4],
    seed = [74],
    global_register = PATH_GR_MNIST_LOCAL,
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/mnist',
    verbose = True
)