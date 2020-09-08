import argparse
from typing import List

from joblib import Parallel, delayed

from scripts.ssc.TopoAE_ext.config_libraries.swissroll import swissroll_run1
from src.models.TopoAE_WitnessComplex.config import ConfigGrid_TopoAE_ext
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default=1, help="Number of threads", type=int)
    parser.add_argument("--configs", default=swissroll_run1, help="Array of config grids", type=List[ConfigGrid_TopoAE_ext])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_input()
    Parallel(n_jobs=args.threads)(delayed(simulator_TopoAE_ext)(config) for config in args.configs)