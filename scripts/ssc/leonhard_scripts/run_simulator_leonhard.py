
from src.models.COREL.train_engine import simulator_COREL


from scripts.ssc.config_library import *


if __name__ == "__main__":
    path = '/cluster/home/schsimo/MT_VAE-TDA/output'

    simulator_COREL(test_run_leonhard,path,verbose= False, data_constant=True)
