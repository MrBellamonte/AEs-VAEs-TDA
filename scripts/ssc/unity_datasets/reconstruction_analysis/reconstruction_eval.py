import os

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import ConvAE_Unity480320


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # get model
    root_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/corgi/rotating/'
    exp1 = 'Unity_RotCorgi-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep200-rlw1-tlw512-mepush_active1-k2-rmax10-seed1-5bedde0c'
    exp2 = 'Unity_RotCorgi-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep200-rlw1-tlw0-mepush_active1-k1-rmax10-seed1-6cef89ab'
    exp3 = 'Unity_RotCorgi-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep200-rlw1-tlw512-mepush_active1-k1-rmax10-seed1-6e771f83'
    exp4 = 'Unity_RotCorgi-version2-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep200-rlw1-tlw512-mepush_active1-k1-rmax10-seed1-bb410e8f'

    exp5 = 'Unity_RotCorgi-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep500-rlw1-tlw512-mepush_active1-k2-rmax10-seed1-9048549c'

    root_path2 = '/Users/simons/MT_data/sync/euler_sync_scratch/output/corgi/corgi_30_std/'
    exp1_2 = 'Unity_RotCorgi-version5-landmarksTrue-seed1-ConvAE_Unity480320-default-lr1_100-bs30-nep5000-rlw1-tlw1024-mepush_active1-k2-rmax10-seed1-a635373e'

    exp1_3 = 'Unity_RotCorgi-version6-landmarksTrue-seed1-ConvAE_Unity480320-default-lr1_100-bs60-nep5000-rlw1-tlw8192-mepush_active1-k2-rmax10-seed1-da2ac592'
    root_path3 = '/Users/simons/MT_data/sync/euler_sync_scratch/output/corgi/corgi_60_semi/'

    path_source = os.path.join(root_path3,exp1_3)


    autoencoder = ConvAE_Unity480320()
    model = WitnessComplexAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(path_source, 'model_state.pth'))
    model.load_state_dict(state_dict)
    model.eval()


    # get latents
    latents = pd.read_csv(os.path.join(path_source, 'train_latents.csv'))

    latents_temp = latents[0:1]
    # get reconstructed images
    labels = latents['labels']
    latents_tensor = torch.tensor(latents[['0','1']].values)
    print(latents_tensor.shape)

    x_hat = model.decode(latents_tensor.float())


    #x_hat = model.decode(torch.tensor(latents_temp[:][['0', '1']].values).float())
    trans = transforms.ToPILImage()


    for i in range(12):
        ii = 20*i
        plt.imshow(trans(x_hat[ii][:][:][:]))
        plt.savefig(os.path.join(path_source,'{}deg.pdf'.format(ii)))
        plt.show()
