"""Module to train a model with a dataset configuration."""
import json
import operator
import os
import statistics

import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances
from torch.utils.data import TensorDataset

from src.datasets.splitting import split_validation
from src.evaluation.eval import Multi_Evaluation
from src.models.COREL.eval_engine import get_latentspace_representation
from src.train_pipeline.callbacks.callbacks import (
    Callback,
    SaveLatentRepresentation)
from src.train_pipeline.training import TrainingLoop
from src.train_pipeline.callbacks.callback_train import LogDatasetLoss, LogTrainingLoss
from src.utils.dict_utils import avg_array_in_dict, default
from src.utils.plots import plot_losses, plot_2Dscatter, plot_distcomp_Z_manifold


class NewlineCallback(Callback):
    """Add newline between epochs for better readability."""

    def on_epoch_end(self, **kwargs):
        print()


def train(model, data_train, data_test, config, device, quiet, val_size, _seed, _rnd, _run, rundir):
    """Sacred wrapped function to run training of model."""

    try:
        os.makedirs(rundir)
    except:
        pass

    train_dataset, validation_dataset = split_validation(
        data_train, val_size, _rnd)
    test_dataset = data_test

    callbacks = [
        LogTrainingLoss(_run, print_progress=operator.not_(quiet)),
        LogDatasetLoss('validation', validation_dataset, _run,
                       method_args=config.method_args,
                       print_progress=operator.not_(quiet), batch_size=config.batch_size,
                       early_stopping=config.early_stopping, save_path=rundir,
                       device=device),
        LogDatasetLoss('testing', test_dataset, _run, method_args=config.method_args,
                       print_progress=operator.not_(quiet),
                       batch_size=config.batch_size, device=device),
    ]

    if quiet:
        pass
    else:
        callbacks.append(NewlineCallback())

    # If we are logging this run save reconstruction images
    if rundir is not None:
        # if hasattr(train_dataset, 'inverse_normalization'):
        #     # We have image data so we can visualize reconstructed images
        #     callbacks.append(SaveReconstructedImages(rundir))
        if config.eval.online_visualization:
            callbacks.append(
                SaveLatentRepresentation(
                    test_dataset, rundir, batch_size=64, device=device)
            )

    training_loop = TrainingLoop(
        model, train_dataset, config.n_epochs, config.batch_size, config.learning_rate,
        config.method_args, config.weight_decay, device, callbacks, verbose=operator.not_(quiet))

    # Run training
    epoch, run_times_epoch = training_loop()

    if rundir:
        # Save model state (and entire model)
        if not quiet:
            print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)
    model.eval()

    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    loss_averages = {
        key: value for key, value in logged_averages.items() if 'loss' in key
    }
    loss_stds = {
        key: value for key, value in logged_stds.items() if 'loss' in key
    }
    metric_averages = {
        key: value for key, value in logged_averages.items() if 'metric' in key
    }
    metric_stds = {
        key: value for key, value in logged_stds.items() if 'metric' in key
    }

    if rundir:
        plot_losses(
            loss_averages,
            loss_stds,
            save_file=os.path.join(rundir, 'loss.pdf'),
        )
        plot_losses(
            metric_averages,
            metric_stds,
            save_file=os.path.join(rundir, 'metrics.pdf'),
            pairs_axes=True
        )

    result = {
        key: values[-1] for key, values in logged_averages.items()
    }

    if config.eval.active:
        evaluate_on = config.eval.evaluate_on
        if evaluate_on == 'validation':
            selected_dataset = validation_dataset
        else:
            selected_dataset = test_dataset


        dataloader_eval = torch.utils.data.DataLoader(
            selected_dataset, batch_size=config.batch_size, pin_memory=True,
            drop_last=False)

        X_eval, Y_eval, Z_eval = get_latentspace_representation(model, dataloader_eval,
                                                                device=device)

        if config.eval.eval_manifold:
            # sample true manifold
            if evaluate_on == 'validation':
                manifold_eval_train = True
            else:
                manifold_eval_train = False
            try:
                dataset = config.dataset
                Z_manifold, X_transformed, labels = dataset.sample_manifold(
                    **config.sampling_kwargs, train=manifold_eval_train)

                dataset_test = TensorDataset(torch.Tensor(X_transformed), torch.Tensor(labels))
                dataloader_eval = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size,
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

                pwd_Z = pairwise_distances(Z_eval, Z_eval, n_jobs=1)
                pwd_Ztrue = pairwise_distances(Z_manifold, Z_manifold, n_jobs=1)

                # normalize distances
                pairwise_distances_manifold = (pwd_Ztrue-pwd_Ztrue.min())/(pwd_Ztrue.max()-pwd_Ztrue.min())
                pairwise_distances_Z = (pwd_Z-pwd_Z.min())/(pwd_Z.max()-pwd_Z.min())



                # save comparison fig
                plot_distcomp_Z_manifold(Z_manifold=Z_manifold, Z_latent=Z_eval,
                                         pwd_manifold=pairwise_distances_manifold,
                                         pwd_Z=pairwise_distances_Z, labels=labels,
                                         path_to_save=rundir, name='manifold_Z_distcomp',
                                         fontsize=24, show=False)

                rmse_manifold = (np.square(pairwise_distances_manifold - pairwise_distances_Z)).mean()
                result.update(dict(rmse_manifold_Z=rmse_manifold))
            except AttributeError as err:
                print(err)
                print('Manifold not evaluated!')

        if rundir and config.eval.save_eval_latent:
            df = pd.DataFrame(Z_eval)
            df['labels'] = Y_eval
            df.to_csv(os.path.join(rundir, 'latents.csv'), index=False)
            np.savez(
                os.path.join(rundir, 'latents.npz'),
                latents=Y_eval, labels=Z_eval
            )
            plot_2Dscatter(Z_eval, Y_eval, path_to_save=os.path.join(
                rundir, 'test_latent_visualization.pdf'), title=None, show=False)

        if rundir and config.eval.save_train_latent:
            dataloader_train = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.batch_size, pin_memory=True,
                drop_last=False
            )
            X_train, Y_train, Z_train = get_latentspace_representation(model, dataloader_train,
                                                                       device=device)

            df = pd.DataFrame(Z_train)
            df['labels'] = Y_train
            df.to_csv(os.path.join(rundir, 'train_latents.csv'), index=False)
            np.savez(
                os.path.join(rundir, 'latents.npz'),
                latents=Z_train, labels=Y_train
            )
            # Visualize latent space
            plot_2Dscatter(Z_train, Y_train, path_to_save=os.path.join(
                rundir, 'train_latent_visualization.pdf'), title=None, show=False)
        if config.eval.quant_eval:
            ks = list(
                range(config.eval.k_min, config.eval.k_max+config.eval.k_step, config.eval.k_step))

            evaluator = Multi_Evaluation(
                dataloader=dataloader_eval, seed=_seed, model=model)
            ev_result = evaluator.get_multi_evals(
                X_eval, Z_eval, Y_eval, ks=ks)
            prefixed_ev_result = {
                config.eval.evaluate_on+'_'+key: value
                for key, value in ev_result.items()
            }
            result.update(prefixed_ev_result)
            s = json.dumps(result, default=default)
            open(os.path.join(rundir, 'eval_metrics.json'), "w").write(s)

    result_avg = avg_array_in_dict(result)
    result_avg.update({'run_times_epoch': statistics.mean(run_times_epoch)})

    return result_avg
