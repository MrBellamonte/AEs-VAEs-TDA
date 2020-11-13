from scripts.ssc.wc_offline.config_libraries.global_register_definitions import PATH_GR_MNIST_EULER
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll, MNIST, MNIST_offline

mnist_1 = ConfigWC_Grid(
    dataset = [MNIST_offline()],
    sampling_kwargs = [dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    batch_size=[64,128,256,512,1024],
    wc_kwargs=[dict()],
    eval_size=[0.14], # not a lot of data "wasted for bs = 4096 & could go to bs=8192
    n_jobs = [16],
    seed = [838],
    global_register = PATH_GR_MNIST_EULER,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist',
    verbose = True
)

mnist_2 = ConfigWC_Grid(
    dataset = [MNIST_offline()],
    sampling_kwargs = [dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA')],
    batch_size=[64,128,256,512,1024],
    wc_kwargs=[dict()],
    eval_size=[0.14], # not a lot of data "wasted for bs = 4096 & could go to bs=8192
    n_jobs = [6],
    seed = [229],
    global_register = PATH_GR_MNIST_EULER,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/mnist',
    verbose = True
)
