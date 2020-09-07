from joblib import Parallel, delayed

from scripts.ssc.TopoAE_ext.config_libraries.swissroll import swissroll_run1
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":

    Parallel(n_jobs=8)(delayed(simulator_TopoAE_ext)(config) for config in swissroll_run1)