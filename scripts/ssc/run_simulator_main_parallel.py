import argparse

import importlib

from joblib import Parallel, delayed

import scripts

from src.models.TopoAE.train_engine import simulator_TopoAE
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model",
                        default='topoae',
                        help="model to run simulation", type=str)
    parser.add_argument('-n', "--n_jobs",
                        default=1,
                        help="number of parallel processes", type=int)
    parser.add_argument('-c', "--configs",
                        default="swissroll.swissroll_testing_parallel",
                        help="configgrid to run simulation", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    if args.model == 'topoae':
        conifg_srt = 'scripts.ssc.TopoAE.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE)(config) for config in configs)
    elif args.model == 'topoae_ext':
        conifg_srt = 'scripts.ssc.TopoAE_ext.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE_ext)(config) for config in configs)


    else:
        raise ValueError("Model {} not defined.".format(args.model))
