import importlib
import json
import os
import statistics

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.evaluation.config import ConfigEval
from src.evaluation.eval import Multi_Evaluation
from src.evaluation.measures import pairwise_distances
from src.models.COREL.eval_engine import get_latentspace_representation
from src.models.WitnessComplexAE.train_engine import COLS_DF_RESULT
from src.models.WitnessComplexAE.wc_ae import WitnessComplexAutoencoder
from src.utils.dict_utils import avg_array_in_dict, default
from src.utils.plots import plot_distcomp_Z_manifold, plot_2Dscatter



def offline_eval_WAE(exp_dir,ConfigEval):


    exp = 'MNIST_offline-seed838-ConvAE_MNIST_SMALL-default-lr1_100-bs1024-nep1-rlw1-tlw1-mepush_active1-k1-rmax10-seed838-40c8140a'
    run_dir = os.path.join(exp_dir,exp)
    #json_file = json.load(os.path.join(exp_dir,'config.json'))

    with open(os.path.join(run_dir,'config.json'), 'r') as f:
        json_file = json.load(f)

    config = json_file['config']

    data_set_str = config['dataset']['py/object']
    mod_name, dataset_name = data_set_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    dataset = getattr(mod, dataset_name)

    dataset = dataset()
    X_test, y_test = dataset.sample(**config['sampling_kwargs'], train=False)
    selected_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    X_train, y_train = dataset.sample(**config['sampling_kwargs'], train=True)
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    model_str = config['model_class']['py/type']
    mod_name2, model_name = model_str.rsplit('.', 1)
    mod2 = importlib.import_module(mod_name2)
    autoencoder = getattr(mod2, model_name)


    autoencoder = autoencoder(**config['model_kwargs'])
    model = WitnessComplexAutoencoder(autoencoder)
    state_dict = torch.load(os.path.join(run_dir, 'model_state.pth'))
    model.load_state_dict(state_dict)
    model.eval()


    dataloader_eval = torch.utils.data.DataLoader(
        selected_dataset, batch_size=config['batch_size'], pin_memory=True,
        drop_last=False)

    X_eval, Y_eval, Z_eval = get_latentspace_representation(model, dataloader_eval,
                                                            device='cpu')

    result = dict()
    if evalconfig.eval_manifold:
        # sample true manifold
        manifold_eval_train = False
        try:
            Z_manifold, X_transformed, labels = dataset.sample_manifold(
                **config['sampling_kwargs'], train=manifold_eval_train)

            dataset_test = TensorDataset(torch.Tensor(X_transformed), torch.Tensor(labels))
            dataloader_eval = torch.utils.data.DataLoader(dataset_test,
                                                          batch_size=config['batch_size'],
                                                          pin_memory=True, drop_last=False)
            X_eval, Y_eval, Z_latent = get_latentspace_representation(model, dataloader_eval,
                                                                      device='cpu')

            Z_manifold[:, 0] = (Z_manifold[:, 0]-Z_manifold[:, 0].min())/(
                    Z_manifold[:, 0].max()-Z_manifold[:, 0].min())
            Z_manifold[:, 1] = (Z_manifold[:, 1]-Z_manifold[:, 1].min())/(
                    Z_manifold[:, 1].max()-Z_manifold[:, 1].min())
            Z_latent[:, 0] = (Z_latent[:, 0]-Z_latent[:, 0].min())/(
                    Z_latent[:, 0].max()-Z_latent[:, 0].min())
            Z_latent[:, 1] = (Z_latent[:, 1]-Z_latent[:, 1].min())/(
                    Z_latent[:, 1].max()-Z_latent[:, 1].min())

            pwd_Z = pairwise_distances(Z_latent, Z_latent, n_jobs=1)
            pwd_Ztrue = pairwise_distances(Z_manifold, Z_manifold, n_jobs=1)

            # normalize distances
            pairwise_distances_manifold = (pwd_Ztrue-pwd_Ztrue.min())/(
                        pwd_Ztrue.max()-pwd_Ztrue.min())
            pairwise_distances_Z = (pwd_Z-pwd_Z.min())/(pwd_Z.max()-pwd_Z.min())

            # save comparison fig
            plot_distcomp_Z_manifold(Z_manifold=Z_manifold, Z_latent=Z_latent,
                                     pwd_manifold=pairwise_distances_manifold,
                                     pwd_Z=pairwise_distances_Z, labels=labels,
                                     path_to_save=run_dir, name='manifold_Z_distcomp',
                                     fontsize=24, show=False)

            rmse_manifold = (np.square(pairwise_distances_manifold-pairwise_distances_Z)).mean()
            result.update(dict(rmse_manifold_Z=rmse_manifold))
        except AttributeError as err:
            print(err)
            print('Manifold not evaluated!')

    if run_dir and evalconfig.save_eval_latent:
        df = pd.DataFrame(Z_eval)
        df['labels'] = Y_eval
        df.to_csv(os.path.join(run_dir, 'latents.csv'), index=False)
        np.savez(
            os.path.join(run_dir, 'latents.npz'),
            latents=Y_eval, labels=Z_eval
        )
        plot_2Dscatter(Z_eval, Y_eval, path_to_save=os.path.join(
            run_dir, 'test_latent_visualization.pdf'), title=None, show=False)

    if run_dir and evalconfig.save_train_latent:
        dataloader_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], pin_memory=True,
            drop_last=False
        )
        X_train, Y_train, Z_train = get_latentspace_representation(model, dataloader_train,
                                                                   device='cpu')

        df = pd.DataFrame(Z_train)
        df['labels'] = Y_train
        df.to_csv(os.path.join(run_dir, 'train_latents.csv'), index=False)
        np.savez(
            os.path.join(run_dir, 'latents.npz'),
            latents=Z_train, labels=Y_train
        )
        # Visualize latent space
        plot_2Dscatter(Z_train, Y_train, path_to_save=os.path.join(
            run_dir, 'train_latent_visualization.pdf'), title=None, show=False)
    if evalconfig.quant_eval:
        ks = list(
            range(evalconfig.k_min, evalconfig.k_max+evalconfig.k_step, evalconfig.k_step))

        evaluator = Multi_Evaluation(
            dataloader=dataloader_eval, seed=config['seed'], model=model)
        ev_result = evaluator.get_multi_evals(
            X_eval, Z_eval, Y_eval, ks=ks)
        prefixed_ev_result = {
            evalconfig.evaluate_on+'_'+key: value
            for key, value in ev_result.items()
        }
        result.update(prefixed_ev_result)
        s = json.dumps(result, default=default)
        open(os.path.join(run_dir, 'eval_metrics.json'), "w").write(s)

    result_avg = avg_array_in_dict(result)

    df = pd.DataFrame.from_dict(result_avg, orient='index').reset_index()
    df.columns = ['metric', 'value']

    id_dict = dict(
        uid = exp,
        learning_rate=config['learning_rate'],
        batch_size = config['batch_size'],
        top_loss_weight = config['top_loss_weight'],
        n_epochs=config['n_epochs'],
        weight_decay=config['weight_decay'],
        early_stopping=config['early_stopping'],
        rec_loss_weight=config['rec_loss_weight']
    )
    for key, value in id_dict.items():
        df[key] = value
    df.set_index('uid')

    df = df[COLS_DF_RESULT]
    #
    df.to_csv(os.path.join(exp_dir, 'eval_metrics_all.csv'), mode='a', header=False)


    pass


if __name__ == "__main__":
    evalconfig = ConfigEval(
        active=False,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        k_min=5,
        k_max=45,
        k_step=5)

    exp_dir = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/mnist_precomputed_2'

    subfolders = [f.path for f in os.scandir(exp_dir) if (f.is_dir() and f)]
    print([x[1] for x in os.walk(exp_dir)])