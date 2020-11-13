from joblib import delayed, Parallel

from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.swissroll import swissroll_h22_list
from scripts.ssc.TopoAE_ext.config_libraries.swissroll import swissroll_testing
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext

configs = [1,2,3,4,5,6,7,8,9,10,11,12,13]

def func(x):
    print(x**2)


for config in swissroll_testing.configs_from_grid():
    simulator_TopoAE_ext(config)

#Parallel(n_jobs=1)(delayed(simulator_TopoAE_ext)(config) for config in configs)

