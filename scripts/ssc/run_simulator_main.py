import argparse

import importlib
from typing import List

import torch

from src.competitors.train_engine import simulator_competitor
from src.data_preprocessing.witness_complex_offline.compute_wc import compute_wc_multiple
from src.models.TopoAE.config import ConfigTopoAE, ConfigGrid_TopoAE
from src.models.TopoAE.train_engine import simulator_TopoAE
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE, ConfigWCAE
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext
from src.utils.config_utils import get_configs


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
        conifg_srt = 'scripts.ssc.models.TopoAE.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        configs = get_configs(configs, ConfigTopoAE, ConfigGrid_TopoAE)
        for config in configs:
            simulator_TopoAE(configs)
    elif args.model == 'topoae_ext':
        conifg_srt = 'scripts.ssc.models.TopoAE_ext.config_libraries.'+args.configs
        mod_name, config_name = conifg_srt.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        configs = getattr(mod, config_name)

        configs = get_configs(configs, ConfigWCAE, ConfigGrid_WCAE)
        for config in configs:
            simulator_TopoAE_ext(config)

    elif args.model == 'competitor':
        conifg_srt = 'scripts.ssc.models.Competitors.config_libraries.'+args.configs
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
