import os

import torch
import seaborn as sns

from src.datasets.datasets import MNIST_offline
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import DeepAE_MNIST
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    path_norm2 = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/retrain_examples/1_'
    exp_norm2 = 'Unity_RotOpenAI-seed1-ConvAE_Unity480320-default-lr1_100-bs180-nep1000-rlw1-tlw1-mepush_active9_8-k2-rmax10-seed1-42e61867'


    wae_2d_kl01 = '/Users/simons/MT_data/eval_data/MNIST_FINAL/WAE/kl01_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_10-bs64-nep1000-rlw1-tlw1_16-mepush_active9_8-k8-rmax10-seed838-565f4980'
    wae_2d_std = '/Users/simons/MT_data/eval_data/MNIST_FINAL/WAE/std_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_10-bs64-nep1000-rlw1-tlw1_16-mepush_active1-k4-rmax10-seed838-89a223af'
    wae_2d_cont = '/Users/simons/MT_data/eval_data/MNIST_FINAL/WAE/cont_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_10-bs64-nep1000-rlw1-tlw1_64-mepush_active1-k16-rmax10-seed838-36a19702'
    wae_2d_rec = '/Users/simons/MT_data/eval_data/MNIST_FINAL/WAE/rec_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_1000-bs1024-nep1000-rlw1-tlw1_256-mepush_active9_8-k1-rmax10-seed838-51c7d219'


    for path_exp in [wae_2d_kl01, wae_2d_std, wae_2d_cont, wae_2d_rec]:
        # get model
        autoencoder = DeepAE_MNIST()

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
        data, labels = dataset.sample(train = False)

        z = model.encode(torch.Tensor(data).float())

        plot_2Dscatter(z.detach().numpy(), labels, path_to_save=os.path.join(
            path_exp, '{}_latent_visualization.pdf'.format('final')), title=None, show=False,
                       palette='custom2')
