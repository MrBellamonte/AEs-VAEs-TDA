import argparse
import importlib

from joblib import Parallel, delayed

from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--configs", default='swissroll.swissroll_testing_euler_parallel', help="Array of config grids", type = str)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_input()
    conifg_srt = 'scripts.ssc.TopoAE_ext.config_libraries.' + args.configs
    mod_name, config_name = conifg_srt.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    configs = getattr(mod, config_name)
    Parallel(n_jobs=len(configs))(delayed(simulator_TopoAE_ext)(config) for config in configs)