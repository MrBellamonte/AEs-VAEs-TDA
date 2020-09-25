"""Module to train a model with a dataset configuration."""
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.measures_optimized import MeasureCalculator
from src.utils.dict_utils import avg_array_in_dict, default
from src.utils.plots import plot_2Dscatter


def eval(result,X,Z,Y,rundir,config,train = True):

    if train:
        name_prefix = 'train'
        save_latent = config.eval.save_train_latent
    else:
        name_prefix = 'test'
        save_latent = config.eval.save_eval_latent

    if rundir and save_latent:
        df = pd.DataFrame(Z)
        df['labels'] = Y
        df.to_csv(os.path.join(rundir, '{}_latents.csv'.format(name_prefix)), index=False)
        np.savez(
            os.path.join(rundir, '{}_latents.npz'.format(name_prefix)),
            latents=Z, labels=Y
        )
        plot_2Dscatter(Z, Y, path_to_save=os.path.join(
            rundir, '{}_latent_visualization.pdf'.format(name_prefix)), title=None, show=False)

    ks = list(range(config.eval.k_min, config.eval.k_max+config.eval.k_step, config.eval.k_step))

    calc = MeasureCalculator(X, Z, max(ks))

    indep_measures = calc.compute_k_independent_measures()
    dep_measures = calc.compute_measures_for_ks(ks)
    mean_dep_measures = {
        'mean_'+key: values.mean() for key, values in dep_measures.items()
    }

    ev_result = {
        key: value for key, value in
        itertools.chain(indep_measures.items(), dep_measures.items(),
                        mean_dep_measures.items())
    }

    prefixed_ev_result = {
        name_prefix+'_'+key: value
        for key, value in ev_result.items()
    }
    result.update(prefixed_ev_result)
    s = json.dumps(result, default=default)
    open(os.path.join(rundir, '{}_eval_metrics.json'.format(name_prefix)), "w").write(s)

    return avg_array_in_dict(result)



def train_comp(model, data_train, data_test, config, quiet,val_size, _seed, _rnd, _run, rundir):
    """Sacred wrapped function to run training of model."""

    try:
        os.makedirs(rundir)
    except:
        pass

    # include split for fair comparison....
    X_train, X_val, y_train, y_val, = train_test_split(data_train[0], data_train[1], test_size = val_size, random_state = _seed)
    test_dataset = data_test

    if not quiet:
        print('Train model...')
    Z_train, y_train = model.get_latent_train(X_train,y_train)

    result = model.eval()
    if not quiet:
        print('Evaluate model on training data...')
    result = eval(result, X_train, Z_train, y_train, rundir, config, train=True)

    if model.test_eval:
        if not quiet:
            print('Evaluate model on test data...')
        Z_test, y_test = model.get_latent_train(test_dataset[0],test_dataset[1])
        result = eval(result, test_dataset[0], Z_test, y_test, rundir, config, train=False)

    return result

