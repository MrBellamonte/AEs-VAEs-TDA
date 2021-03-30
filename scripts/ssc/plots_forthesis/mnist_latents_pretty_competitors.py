import os

import torch
import seaborn as sns
import numpy as np
import pandas as pd

from src.datasets.datasets import MNIST_offline
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import DeepAE_MNIST
from src.utils.plots import plot_2Dscatter

if __name__ == "__main__":
    tsne_rmsez_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_tsne/MNIST_offline-n_samples10000-tSNE--n_jobs1-perplexity5-seed1318-b2b38aea'

    data = pd.read_csv(os.path.join(tsne_rmsez_path, 'train_latents.csv'))

    latents = data[['0', '1']][:].to_numpy()
    labels = data['labels'].tolist()
    #labels = data[['labels']][:].tolist()
    plot_2Dscatter(latents, labels, path_to_save=os.path.join(
        tsne_rmsez_path, '{}_latent_visualization.pdf'.format('final')), title=None, show=False,
                   palette='custom2')
