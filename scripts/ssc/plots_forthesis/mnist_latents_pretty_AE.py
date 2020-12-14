import os

import torch
import seaborn as sns

from src.datasets.datasets import MNIST_offline
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import DeepAE_MNIST
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    ae_2d_kl01 = '/Users/simons/MT_data/eval_data/MNIST_FINAL/AE/kl01_minimizer/MNIST_offline-seed1988-DeepAE_MNIST-default-lr1_10-bs64-nep1000-seed1988-1bfdf21c'
    ae_2d_std = '/Users/simons/MT_data/eval_data/MNIST_FINAL/AE/std_minimizer/MNIST_offline-seed579-DeepAE_MNIST-default-lr1_1000-bs512-nep1000-seed579-c8f54a66'
    ae_2d_cont = '/Users/simons/MT_data/eval_data/MNIST_FINAL/AE/cont_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_1000-bs128-nep1000-seed838-af947653'
    ae_2d_rec = '/Users/simons/MT_data/eval_data/MNIST_FINAL/AE/rec_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_1000-bs128-nep1000-seed838-af947653'


    for path_exp in [ae_2d_kl01, ae_2d_std, ae_2d_cont, ae_2d_rec]:
        # get model
        autoencoder = DeepAE_MNIST()

        model = autoencoder
        state_dict = torch.load(os.path.join(path_exp, 'model_state.pth'),
                                map_location=torch.device('cpu'))


        model.load_state_dict(state_dict)
        model.eval()

        dataset = MNIST_offline()
        data, labels = dataset.sample(train = False)

        z = model.encode(torch.Tensor(data).float())

        plot_2Dscatter(z.detach().numpy(), labels, path_to_save=os.path.join(
            path_exp, '{}_latent_visualization.pdf'.format('final')), title=None, show=False,
                       palette='custom2')
