# stdiso
import os

import torch
import numpy as np

from src.datasets.datasets import MNIST_offline
from src.models.TopoAE.approx_based import TopologicallyRegularizedAutoencoder
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import DeepAE_MNIST_3D

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
path_norm2 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/retrain_examples/1_'
exp_norm2 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-42e61867'


topoae_path_iso = '/Users/simons/MT_data/eval_data/MNIST3D_FINAL/stdiso/TopoAE/MNIST_offline-DeepAE_MNIST_3D-default-lr1_10-bs64-nep1000-rlw1-tlw1_32-seed838-a6d2cf0e'
ae_path_iso = '/Users/simons/MT_data/eval_data/MNIST3D_FINAL/stdiso/AE/MNIST_offline-seed1988-DeepAE_MNIST_3D-default-lr1_1000-bs128-nep1000-seed1988-cd903971'

# kldiv
topoae_path_kl01 = '/Users/simons/MT_data/eval_data/MNIST3D_FINAL/kl01/TopoAE/MNIST_offline-seed579-DeepAE_MNIST_3D-default-lr1_10-bs256-nep1000-rlw1-tlw2-seed579-f2e679b8'
ae_path_kl01 = '/Users/simons/MT_data/eval_data/MNIST3D_FINAL/kl01/AE/MNIST_offline-seed579-DeepAE_MNIST_3D-default-lr1_10-bs128-nep1000-seed579-ac76b83f'

m_topoae = 'TopoAE'
m_ae = 'AE'
m_wae = 'WAE'

MODEL = m_ae
path_exp = ae_path_kl01


if path_exp == topoae_path_iso:
    name = 'topoae_path_iso'
elif path_exp == ae_path_iso:
    name = 'ae_path_iso'
elif path_exp == topoae_path_kl01:
    name = 'topoae_path_kl01'
elif path_exp == ae_path_kl01:
    name = 'ae_path_kl01'
else:
    ValueError

#get model
if MODEL == m_ae:
    autoencoder = DeepAE_MNIST_3D()

    model = autoencoder
    state_dict = torch.load(os.path.join(path_exp, 'model_state.pth'),
                            map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model.eval()

    dataset = MNIST_offline()
    data, labels = dataset.sample(train=False)

    z = model.encode(torch.Tensor(data).float())
elif MODEL == m_topoae:
    autoencoder = DeepAE_MNIST_3D()
    model = TopologicallyRegularizedAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(path_exp, 'model_state.pth'),
                            map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model.eval()

    dataset = MNIST_offline()
    data, labels = dataset.sample(train=False)

    z = model.encode(torch.Tensor(data).float())

elif MODEL == m_wae:
    autoencoder = DeepAE_MNIST_3D()
    model = WitnessComplexAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(path_exp, 'model_state.pth'),
                            map_location=torch.device('cpu'))
    state_dict2 = torch.load(
        os.path.join(os.path.join(path_norm2, exp_norm2), 'model_state.pth'))
    if 'latent' not in state_dict:
        state_dict['latent_norm'] = state_dict2['latent_norm']*0.1

    print('passed')

    model.load_state_dict(state_dict)
    model.eval()

    dataset = MNIST_offline()
    data, labels = dataset.sample(train=False)

    z = model.encode(torch.Tensor(data).float())
else:
    ValueError
    
np.save(os.path.join('/Users/simons/MT_data/eval_data/MNIST3D_FINAL/latents','labels{}'.format(str(name))),labels)
np.save(os.path.join('/Users/simons/MT_data/eval_data/MNIST3D_FINAL/latents','{}'.format(str(name))),z.detach().numpy())





