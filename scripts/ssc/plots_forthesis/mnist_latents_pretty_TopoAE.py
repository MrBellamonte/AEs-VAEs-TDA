import os

import torch
import seaborn as sns

from src.datasets.datasets import MNIST_offline
from src.models.TopoAE.approx_based import TopologicallyRegularizedAutoencoder
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import DeepAE_MNIST
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    topoae_2d_kl01 = '/Users/simons/MT_data/eval_data/MNIST_FINAL/TopoAE/kl01_minimizer/MNIST_offline-seed1988-DeepAE_MNIST-default-lr1_10-bs256-nep1000-rlw1-tlw4-seed1988-d0628438'
    topoae_2d_std = '/Users/simons/MT_data/eval_data/MNIST_FINAL/TopoAE/std_minimizer/MNIST_offline-seed1988-DeepAE_MNIST-default-lr1_1000-bs64-nep1000-rlw1-tlw1_256-seed1988-1ae25b75'
    topoae_2d_cont = '/Users/simons/MT_data/eval_data/MNIST_FINAL/TopoAE/cont_minimizer/MNIST_offline-seed579-DeepAE_MNIST-default-lr1_1000-bs128-nep1000-rlw1-tlw1_128-seed579-3c78a835'
    topoae_2d_rec = '/Users/simons/MT_data/eval_data/MNIST_FINAL/TopoAE/rec_minimizer/MNIST_offline-seed838-DeepAE_MNIST-default-lr1_1000-bs256-nep1000-rlw1-tlw1_16-seed838-67de8d97'


    for path_exp in [topoae_2d_kl01, topoae_2d_std, topoae_2d_cont, topoae_2d_rec]:
        # get model
        autoencoder = DeepAE_MNIST()

        model = TopologicallyRegularizedAutoencoder(autoencoder)
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
