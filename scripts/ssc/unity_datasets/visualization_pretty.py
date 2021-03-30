# get data
import os
import time

import torch

from src.datasets.datasets import Unity_XYTransOpenAI
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import ConvAE_Unity480320
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    path_norm2 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/retrain_examples/1_'
    exp_norm2 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-42e61867'



    MODEL_wae1_1 =  '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_l_newpers_1/Unity_XYTransOpenAI-versionxy_trans_l_newpers-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep5000-rlw1-tlw2-mepush_active1-k4-rmax10-seed2-fc8ff431'
    PATHSAVE_wae1_1 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/unity_final_vis/wae_prev'

    MODEL_wae1_2 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_l_newpers_1/Unity_XYTransOpenAI-versionxy_trans_l_newpers-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep5000-rlw1-tlw8-mepush_active9_8-k4-rmax10-seed2-e037bb0e'
    PATHSAVE_wae1_2 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/unity_final_vis/wae_prev2_datafinal'

    MODEL_wae2_1 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_final/goood/Unity_XYTransOpenAI-versionxy_trans_final-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep2000-rlw1-tlw16-mepush_active9_8-k3-rmax10-seed2-9b04fc86'
    PATHSAVE_wae2_1 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/unity_final_vis/wae_final_datafinal'

    MODEL_vae2_1 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_final_notopo/Unity_XYTransOpenAI-versionxy_trans_final-seed838-ConvAE_Unity480320-default-lr1_1000-bs200-nep10000-seed838-8653b77f'
    PATHSAVE_vae2_1 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/unity_final_vis/vae_final_datafinal'

    MODEL_vae1_1 = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_l_newpers_notopo/Unity_XYTransOpenAI-versionxy_trans_l_newpers-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep15000-rlw1-tlw0-mepush_active1-k1-rmax10-seed2-c7ac9997'
    PATHSAVE_vae1_1 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/unity_final_vis/vae_prev'

    path_model = MODEL_wae1_2
    path_save = PATHSAVE_wae1_2


    # prepare data
    position_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_l_newpers/position.pt'
    #position_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_final/position.pt'
    position = torch.load(position_path)
    x_position = position[:, 0]
    y_position = position[:, 1]

    images_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_l_newpers/images.pt'
    #images_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_final/images.pt'
    images = torch.load(images_path)


    # get model
    autoencoder = ConvAE_Unity480320()

    if path_model == MODEL_vae2_1:
        state_dict = torch.load(os.path.join(path_model, 'model_state.pth'),map_location=torch.device('cpu'))
        model = autoencoder
        print('Loading Passed')
    else:
        model = WitnessComplexAutoencoder(autoencoder)
        state_dict = torch.load(os.path.join(path_model, 'model_state.pth'),map_location=torch.device('cpu'))

        state_dict2 = torch.load(os.path.join(os.path.join(path_norm2,exp_norm2), 'model_state.pth'))
        if 'latent' not in state_dict:
            state_dict['latent_norm'] = state_dict2['latent_norm'] * 0.1

        print('passed')

    model.load_state_dict(state_dict)
    model.eval()

    z = model.encode(images.float())
    debugging = True

    plot_2Dscatter(z.detach().numpy(), x_position.detach().numpy(), path_to_save=os.path.join(path_save, 'x_vis_{}.pdf'.format(
        time.strftime("%Y%m%d-%H%M%S"))), title=None, show=True,palette='x')
    plot_2Dscatter(z.detach().numpy(), y_position.detach().numpy(), path_to_save=os.path.join(path_save, 'y_vis_{}.pdf'.format(
        time.strftime("%Y%m%d-%H%M%S"))), title=None, show=True,palette='y')

