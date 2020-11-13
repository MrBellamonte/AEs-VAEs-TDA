import argparse

import importlib
import random

from joblib import Parallel, delayed

from src.competitors.config import ConfigGrid_Competitors
from src.competitors.train_engine import simulator_competitor
from src.models.TopoAE.config import ConfigGrid_TopoAE

from src.models.TopoAE.train_engine import simulator_TopoAE
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model",
                        default='WCAE',
                        help="model to run simulation", type=str)
    parser.add_argument('-n', "--n_jobs",
                        default=1,
                        help="number of parallel processes", type=int)
    parser.add_argument('-c', "--configs",
                        default="WCAE_sample_config",
                        help="configgrid to run simulation", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    if args.model == 'topoae':
        conifg_srt = 'scripts.config_library.sample.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_TopoAE):
            configs = configs.configs_from_grid()
        else:
            configs = configs

        if args.n_jobs > 1:
            Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE)(config) for config in configs)
        else:
            for config in configs:
                simulator_TopoAE(config)

    elif args.model == 'WCAE':
        conifg_srt = 'scripts.config_library.sample.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_WCAE):
            configs = configs.configs_from_grid()
        else:
            configs = configs
        if args.n_jobs > 1:
            Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE_ext)(config) for config in configs)
        else:
            for config in configs:
                simulator_TopoAE_ext(config)

    elif args.model == 'competitor':
        conifg_srt = 'scripts.config_library.sample.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_Competitors):
            configs = configs.configs_from_grid()
        else:
            configs = configs

        if args.n_jobs > 1:
            Parallel(n_jobs=args.n_jobs)(delayed(simulator_competitor)(config) for config in configs)
        else:
            for config in configs:
                simulator_competitor(config)

    else:
        raise ValueError("Model {} not defined.".format(args.model))
