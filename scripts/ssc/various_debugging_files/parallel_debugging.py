from joblib import delayed, Parallel

from scripts.ssc.TopoAE_ext.config_libraries.euler_configs.swissroll import swissroll_h22_list

configs = [1,2,3,4,5,6,7,8,9,10,11,12,13]

def func(x):
    print(x**2)




Parallel(n_jobs=2)(delayed(func)(config) for config in configs)

