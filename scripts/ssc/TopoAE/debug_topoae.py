import argparse
import importlib

from joblib import Parallel, delayed

from scripts.ssc.TopoAE.config_libraries.swissroll import swissroll_lle
from src.models.TopoAE.train_engine import simulator_TopoAE

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--configs", default='swissroll.swissroll_testing_euler_parallel', help="Array of config grids", type = str)
    return parser.parse_args()



if __name__ == "__main__":
    simulator_TopoAE(swissroll_lle)
