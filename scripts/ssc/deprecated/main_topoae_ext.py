import argparse
import json

import importlib


import scripts
from scripts.ssc.TopoAE_ext.config_libraries.swissroll import swissroll_testing
from src.models.TopoAE_WitnessComplex.config import ConfigGrid_TopoAE_ext
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--configs", default=scripts.ssc.TopoAE_ext.config_libraries.swissroll.swissroll_testing, help="Array of config grids", type = str)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_input()

    #args.configs = 'scripts.ssc.TopoAE_ext.config_libraries.swissroll.swissroll_testing'
    print(args.configs)

    conifg_srt = 'scripts.ssc.TopoAE_ext.config_libraries.' + args.configs
    mod_name, config_name = conifg_srt.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    configs = getattr(mod, config_name)

    simulator_TopoAE_ext(configs)