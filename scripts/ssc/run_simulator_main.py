import argparse

import importlib
from typing import List

import torch

from src.competitors.train_engine import simulator_competitor
from src.data_preprocessing.witness_complex_offline.compute_wc import compute_wc_multiple
from src.models.TopoAE.train_engine import simulator_TopoAE
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE, ConfigWCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model",
                        default='topoae',
                        help="model to run simulation", type=str)
    parser.add_argument('-c', "--configs",
                        default="swissroll.swissroll_testing",
                        help="configgrid to run simulation", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    if args.model == 'topoae':
        conifg_srt = 'scripts.ssc.TopoAE.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        simulator_TopoAE(configs)
    elif args.model == 'topoae_ext':
        conifg_srt = 'scripts.ssc.TopoAE_ext.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        if isinstance(configs, ConfigGrid_WCAE):
            configs = configs.configs_from_grid()
        elif isinstance(configs, ConfigWCAE):
            configs = [configs]
        elif isinstance(configs, List):
            configs = configs
        else:
            raise ValueError

        for config in configs:
            simulator_TopoAE_ext(config)

    elif args.model == 'competitor':
        conifg_srt = 'scripts.ssc.Competitors.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        simulator_competitor(configs)
    elif args.model == 'wc_offline':
        conifg_srt = 'scripts.ssc.wc_offline.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        compute_wc_multiple(configs)
    else:
        raise ValueError("Model {} not defined.".format(args.model))
