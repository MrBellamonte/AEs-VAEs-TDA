from scripts.ssc.wc_offline.config_libraries.global_register_definitions import PATH_GR_UNITY_LOCAL
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import Unity_XYTransOpenAI

unity_xytrans1 = ConfigWC_Grid(
    dataset = [Unity_XYTransOpenAI(version = 'xy_trans_l_newpers')],
    sampling_kwargs = [dict()],
    batch_size=[200],
    wc_kwargs=[dict()],
    eval_size=[0], # eval set same as train
    n_jobs = [1],
    seed = [1],
    global_register = PATH_GR_UNITY_LOCAL,
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/unity',
    verbose = True
)