import os

import numpy as np
import pandas as pd
import torch
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
    df_path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/multi_k/evaldata_mldl_full_small.csv'

    # set which models to evaluate
    wctopoae_64 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/WCTopoAE/bs64'
    wctopoae_128 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/WCTopoAE/bs128'
    wctopoae_256 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/WCTopoAE/bs256'
    topoae_64 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/TopoAE/bs64'
    topoae_128 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/TopoAE/bs128'
    topoae_256 = '/Users/simons/MT_data/eval_all_analysis/Selection_WP/TopoAE/bs256'

    eval_models_dict = {
        'TopoAE64'  : topoae_64, 'TopoAE128': topoae_128, 'TopoAE256': topoae_256,
        'WCTopoAE64': wctopoae_64, 'WCTopoAE128': wctopoae_128, 'WCTopoAE256': wctopoae_256,

    }
    # eval_models_dict = {
    #    'TopoAE128': topoae_128,
    #     'WCTopoAE128': wctopoae_128,
    #
    # }
    # eval_models_dict = {
    #     'TopoAE64'  : topoae_64
    # }
    # set metrices to evaluate
    #metrics = ['K_min', 'K_max','K_avg','llrme','continuity','trustworthiness']
    #metrics = ['Trust', 'Cont', 'LGD', 'K_min', 'K_max', 'K_avg']

    metrics = ['RRE','Trust','Cont','IsoX','IsoZ','IsoXlist','IsoZlist']

    # sample data
    n_samples = 2560
    dataset = SwissRoll()
    data, labels = dataset.sample(n_samples=n_samples, seed=1)

    for model_name, path in eval_models_dict.items():
        # load WC-AE
        print('START: {}'.format(model_name))
        model_kwargs = dict(input_dim=3, latent_dim=2, size_hidden_layers=[32, 32])
        autoencoder = Autoencoder_MLP_topoae(**model_kwargs)
        model = WitnessComplexAutoencoder(autoencoder)
        state_dict = torch.load(os.path.join(path, 'model_state.pth'))
        model.load_state_dict(state_dict)
        model.eval()

        dataset_test = TensorDataset(torch.Tensor(data), torch.Tensor(labels))
        dataloader_eval = torch.utils.data.DataLoader(dataset_test, batch_size=n_samples,
                                                      pin_memory=True, drop_last=False)

        X_eval, Y_eval, Z_eval = get_latentspace_representation(model, dataloader_eval,
                                                                device='cpu')

        # evaluate for multiple ks, what? -> Cont, Trust, ll-RMSE, K
        ks = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
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
