import os

import torch
import onnx

from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import ConvAE_Unity480320, ConvAE_Unity480320_inference





if __name__ == "__main__":
    root_path2 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/rotating_decay'
    exp2 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active1-k2-rmax10-seed1-4725d9e2'


    root_path_rl = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_l_newpers_1'
    exp_rl = 'Unity_XYTransOpenAI-versionxy_trans_l_newpers-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep5000-rlw1-tlw8-mepush_active9_8-k4-rmax10-seed2-e037bb0e'

    model_path = '/Users/simons/MT_data/sync/leonhard_sync_scratch/xy_trans_final/goood/Unity_XYTransOpenAI-versionxy_trans_final-seed2-ConvAE_Unity480320-default-lr1_100-bs200-nep2000-rlw1-tlw16-mepush_active9_8-k3-rmax10-seed2-9b04fc86'


    autoencoder = ConvAE_Unity480320_inference()
    device = torch.device('cpu')
    model = WitnessComplexAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(model_path, 'model_state.pth'),map_location=device)
    state_dict2 = torch.load(os.path.join(os.path.join(root_path2,exp2), 'model_state.pth'), map_location=device)
    if 'latent' not in state_dict:
        state_dict['latent_norm'] = state_dict2['latent_norm']
    model.load_state_dict(state_dict)

    ae_trained = model.autoencoder



    dir_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/RL_unity/test_model'
    path_to_save = os.path.join(dir_to_save,'wae_9b04fc86.tjm')
    #torch.save(ae_trained, path_to_save)


    dummy_image = torch.zeros(1,3,320,480)

    model = torch.jit.trace(ae_trained, dummy_image)
    torch.jit.save(model, path_to_save)

    # ae_trained.eval()
    # print(ae_trained(dummy_image))
    #
    # torch.onnx.export(ae_trained,                              # model being run
    #                   dummy_image,                       # model dummy input (or a tuple for multiple inputs)
    #                   path_to_save,                  # where to save the model (can be a file or file-like object)
    #                   export_params=True,                 # store the trained parameter weights inside the model file            # the ONNX version to export the model to
    #                   do_constant_folding=True,           # whether to execute constant folding for optimization
    #                   input_names = ['x'],                # the model's input names
    #                   output_names = ['z']                # the model's output names
    #                   )
    #
    #
    #
    # # Load the ONNX model
    # model = onnx.load(path_to_save)