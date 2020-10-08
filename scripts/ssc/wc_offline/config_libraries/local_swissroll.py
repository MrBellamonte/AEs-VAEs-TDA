from scripts.ssc.wc_offline.config_libraries.global_register_definitions import *
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll

swissroll_nonoise = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [36],
    global_register = PATH_GR_SWISSROLL_LOCAL,
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/SwissRoll/nonoise',
    verbose = True
)
