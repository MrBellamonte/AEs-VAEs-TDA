from scripts.ssc.models.vanillaAE.config_libraries.debug import ae_test
from src.models.vanillaAE.train_engine import simulator_VanillaAE

ae_test

simulator_VanillaAE(ae_test.configs_from_grid()[0])