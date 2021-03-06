import os
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms

from src.models.COREL.eval_engine import get_latentspace_representation
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import ConvAE_Unity480320
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/retrain_examples/1_'
    path_source_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai_trans_final/1_transfer'
    # get model
    exp1 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-42e61867'
    exp2 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-5a3ddf79'
    exp3  = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_1000-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k3-rmax10-seed1-c293b4d1'
    exp4 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_1000-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-4e0ac51f'

    root_path_1 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/rotating_retrain'
    exp_1 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_1000-bs180-nep20000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-becee7e8'

    root_path_2 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_l_newpers_1'
    exp_2_1 = 'Unity_XYTransOpenAI-versionxy_trans_l_newpers-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep5000-rlw1-tlw8-mepush_active9_8-k4-rmax10-seed2-e037bb0e'

    root_path_model = root_path_2
    exp_model = exp_2_1

    path_source = os.path.join(root_path_model,exp_model)


    dataloarder_train = torch.load(os.path.join('/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_final', 'dataloader_train.pt'))



    try:
        os.mkdir(os.path.join(path_source,'latents_inproc'))
    except:
        pass

    autoencoder = ConvAE_Unity480320()
    model = WitnessComplexAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(path_source, 'model_state.pth'),map_location=torch.device('cpu'))

    state_dict2 = torch.load(os.path.join(os.path.join(root_path,exp1), 'model_state.pth'))
    if 'latent' not in state_dict:
        state_dict['latent_norm'] = state_dict2['latent_norm'] * 0.1

    model.load_state_dict(state_dict)
    model.eval()


    X_eval, Y_eval, Z_eval = get_latentspace_representation(model, dataloarder_train,
                                                              device='cpu')


    plot_2Dscatter(Z_eval, Y_eval, path_to_save=os.path.join(path_source_save, '{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S"))), title=None, show=True)
