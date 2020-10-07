from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll

test_wcconfig = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560)],
    batch_size=[128],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [4],
    seed = [1],
    global_register = ['/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/wc_global_register.csv'],
    root_path = ['/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/tests'],
    verbose = [True]
)