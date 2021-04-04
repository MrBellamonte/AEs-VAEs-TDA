from scripts.ssc.models.TopoAE_ext.config_libraries.local_configs.swissroll_vae import vae_test
from scripts.ssc.wc_offline.config_libraries.local.mnist import mnist_2

from src.data_preprocessing.witness_complex_offline.compute_wc import compute_wc_multiple
from src.models.WitnessComplexAE.train_engine import simulator_TopoAE_ext
from src.models.variational_autoencoder.varautoencoders import VanillaVAE
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

#imulator_TopoAE_ext(vae_test.configs_from_grid()[0])



compute_wc_multiple(mnist_2)