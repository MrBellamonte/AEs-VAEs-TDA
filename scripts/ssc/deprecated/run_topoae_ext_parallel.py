import argparse

from joblib import Parallel, delayed

from scripts.ssc.TopoAE_ext.config_libraries.swissroll import swissroll_run1
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default=30, help="Number of threads", type=int)
    return parser.parse_args()


if __name__ == "__main__":

    Parallel(n_jobs=4)(delayed(simulator_TopoAE_ext)(config) for config in swissroll_run1)