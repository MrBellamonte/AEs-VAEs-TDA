from scripts.ssc.wc_offline.config_libraries.global_register_definitions import \
    path_gr_swissroll_euler
from src.data_preprocessing.witness_complex_offline.config import ConfigWC_Grid
from src.datasets.datasets import SwissRoll

swissroll_nonoise = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [36, 3851, 2570, 4304, 1935, 7954, 5095, 5310, 1577, 3288],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/nonoise',
    verbose = True
)

swissroll_noise005 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.05)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [6973, 5305, 6233, 1503, 3947, 1425, 3391, 2941, 1218, 7946],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)

swissroll_noise01 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.1)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [9690, 9868,  657, 4677, 5135, 3141, 8411, 2241, 8720, 5825],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)

swissroll_noise015 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.15)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [ 627, 5716, 2673, 9673, 7632, 9794, 7175, 5247, 7839, 8296],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)

swissroll_noise02 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.2)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [9398, 7827, 5541, 5040, 6207,  543, 3152, 1537, 8123, 4668],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)

swissroll_noise025 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.2)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [ 977, 3800, 6451, 6320, 7403,  581, 6299, 2833, 4278, 4090],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)

swissroll_noise03 = ConfigWC_Grid(
    dataset = [SwissRoll()],
    sampling_kwargs = [dict(n_samples = 2560, noise = 0.2)],
    batch_size=[64,128,256,512],
    wc_kwargs=[dict()],
    eval_size=[0.2],
    n_jobs = [2],
    seed = [1200, 4581, 6925, 3134, 4827, 7961, 1451, 7419, 1342, 2782],
    global_register = path_gr_swissroll_euler,
    root_path = '/cluster/home/schsimo/MT/output/WitnessComplexes/SwissRoll/noise',
    verbose = True
)