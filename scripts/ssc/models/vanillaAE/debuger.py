from scripts.ssc.models.vanillaAE.config_libraries.debug import ae_test, unity_test
from src.models.vanillaAE.train_engine import simulator_VanillaAE


simulator_VanillaAE(unity_test.configs_from_grid()[0])