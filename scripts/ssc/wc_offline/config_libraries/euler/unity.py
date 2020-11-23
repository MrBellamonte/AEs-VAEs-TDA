from scripts.ssc.wc_offline.config_libraries.global_register_definitions import PATH_GR_UNITY_EULER
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import Unity_XYTransOpenAI

unity_xytrans_rot = ConfigWC_Grid(
    dataset = [Unity_XYTransOpenAI(version = 'xy_trans_rot')],
    sampling_kwargs = [dict(root_path = '/cluster/scratch/schsimo')],
    batch_size=[300],
    wc_kwargs=[dict()],
    eval_size=[0], # eval set same as train
    n_jobs = [4],
    seed = [1],
    global_register = PATH_GR_UNITY_EULER,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/unity',
    verbose = True
)