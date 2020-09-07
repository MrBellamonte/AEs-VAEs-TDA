from scripts.ssc.TopoAE_ext.config_libraries.swissroll import (
    swissroll_testing_euler)
from src.models.TopoAE_WitnessComplex.train_engine import simulator_TopoAE_ext

if __name__ == "__main__":
    simulator_TopoAE_ext(swissroll_testing_euler)



