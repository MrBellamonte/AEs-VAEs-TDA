import argparse

import importlib
import random

from joblib import Parallel, delayed

from src.competitors.config import Config_Competitors, ConfigGrid_Competitors
from src.competitors.train_engine import simulator_competitor
from src.models.TopoAE.config import ConfigGrid_TopoAE

from src.models.TopoAE.train_engine import simulator_TopoAE
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext
from src.models.vanillaAE.config import Config_VanillaAE, ConfigGrid_VanillaAE
from src.models.vanillaAE.train_engine import simulator_VanillaAE
from src.utils.config_utils import get_configs


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
        conifg_srt = 'scripts.ssc.models.TopoAE.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_TopoAE):
            configs = configs.configs_from_grid()
        else:
            configs = configs

        Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE)(config) for config in configs)
    elif args.model == 'topoae_ext':
        conifg_srt = 'scripts.ssc.models.TopoAE_ext.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_WCAE):
            configs = configs.configs_from_grid()
        else:
            configs = configs
        random.shuffle(configs)
        Parallel(n_jobs=args.n_jobs)(delayed(simulator_TopoAE_ext)(config) for config in configs)
    elif args.model == 'vanilla_ae':
        conifg_srt = 'scripts.ssc.models.vanillaAE.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        configs = get_configs(configs, Config_VanillaAE, ConfigGrid_VanillaAE)
        random.shuffle(configs)
        Parallel(n_jobs=args.n_jobs)(delayed(simulator_VanillaAE)(config) for config in configs)

    elif args.model == 'competitor':
        conifg_srt = 'scripts.ssc.models.Competitors.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        configs = get_configs(configs, Config_Competitors, ConfigGrid_Competitors)
        random.shuffle(configs)
        Parallel(n_jobs=args.n_jobs)(delayed(simulator_competitor)(config) for config in configs)
    else:
        raise ValueError("Model {} not defined.".format(args.model))
