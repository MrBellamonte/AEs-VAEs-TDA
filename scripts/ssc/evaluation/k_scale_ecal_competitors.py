import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from scripts.ssc.evaluation.mldl_copied import CompPerformMetrics
from src.datasets.datasets import SwissRoll
from src.evaluation.eval import Multi_Evaluation
from src.models.COREL.eval_engine import get_latentspace_representation
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae


def update_dict(dict, ks, metric, result):
    for i, k in enumerate(ks):
        dict.update({metric+'_k{}'.format(k): result[metric][i]})

    return dict


if __name__ == "__main__":

    # create df and set path to save
    df_tot = pd.DataFrame()
    df_path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/multi_k/evaldata_mldl_competitors.csv'

    # set which models to evaluate
    UMAP_seed = 887
    UMAP_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/umap_swissroll_2/SwissRoll-n_samples2560-UMAP--n_neighbors32-min_dist1_2-seed887-3db9b673/train_latents.npz'

    tSNE_see = 672
    tSNE_path = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/tsne_swissroll_2/SwissRoll-n_samples2560-tSNE--n_jobs1-perplexity50-seed672-017a5cba/train_latents.npz'

    eval_models_dict = {
        'UMAP'  : UMAP_path,
        'tSNE': tSNE_path
    }
    eval_seeds = {
        'UMAP': UMAP_seed,
        'tSNE': tSNE_see
    }

    metrics = ['RRE','Trust','Cont','IsoX','IsoZ','IsoXlist','IsoZlist']

    # sample data



    for model_name, path in eval_models_dict.items():
        # load WC-AE
        print('START: {}'.format(model_name))

        n_samples = 2560
        dataset = SwissRoll()
        X_eval, labels = dataset.sample(n_samples=n_samples, seed=eval_seeds[model_name])
        X_eval, X_val, y_train, y_val, = train_test_split(X_eval, X_eval,
                                                           test_size=0.2, random_state=eval_seeds[model_name])
        Z_eval = np.load(path)['latents']

        # evaluate for multiple ks, what? -> Cont, Trust, ll-RMSE, K
        ks = [15,30,45]
        #ks = [int(k) for k in np.linspace(15,150,10)]



        # eval = Multi_Evaluation(model=model)
        # ev_result = eval.get_multi_evals(data=X_eval, latent=Z_eval, labels=Y_eval, ks=ks)
        ev_result = CompPerformMetrics(X_eval, Z_eval, ks = ks, dataset='norm')

        print('Done')

        # collect results and save in df.

        d = dict(model=model_name)
        for metric in metrics:
            d = update_dict(d, ks, metric=metric, result=ev_result)



        df = pd.DataFrame({k: [v] for k, v in d.items()})
        df_tot = df_tot.append(df)

    print(df_tot)
    df_tot.to_csv(df_path_to_save)
