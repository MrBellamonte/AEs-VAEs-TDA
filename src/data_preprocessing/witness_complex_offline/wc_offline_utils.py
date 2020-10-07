import os
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from src.data_preprocessing.witness_complex_offline.config import ConfigWC
from src.data_preprocessing.witness_complex_offline.definitions import *
from src.topology.witness_complex import WitnessComplex


def select_witnesses(X):
    '''
    Utility to subsample witnesses
    '''
    # todo implement subsampling
    return X


def compute_wc_offline(config: ConfigWC):
    '''
    Utility to compute distances between landmarks "offline" and save dataloader/distance matrices for later training.
    '''

    torch.manual_seed(config.seed)

    # get data, i.e. sample or load
    dataset = config.dataset
    X, y = dataset.sample(**config.sampling_kwargs, seed=config.seed, train=True)

    # "assign witnesses" -> in the future maybe also through sub-sampling
    X_witnesses = select_witnesses(X, **config.wc_kwargs)

    # split datasets into train and eval
    dataset_tensor = TensorDataset(torch.Tensor(X), torch.Tensor(y))

    train_size = int(config.eval_size*y.size)
    eval_size = int(y.size-train_size)

    train_dataset, validation_dataset = random_split(dataset_tensor, (train_size, eval_size))

    # load batches
    dataloader_train = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True, drop_last=False, shuffle=False
    )
    dataloader_eval = DataLoader(
        validation_dataset, batch_size=config.batch_size, pin_memory=True, drop_last=False,
        shuffle=False
    )

    # compute witness complexes
    landmark_dist_train = torch.zeros(config.batch_size, config.batch_size, len(dataloader_train))
    landmark_dist_eval = torch.zeros(config.batch_size, config.batch_size, len(dataloader_eval))

    # save dataloader, landmarks distances for eval and train
    for batch_i, (X_batch, label_barch) in enumerate(dataloader_train):
        witness_complex_train = WitnessComplex(landmarks=X_batch, witnesses=X_witnesses,
                                               n_jobs=config.n_jobs)
        witness_complex_train.compute_metric_optimized(n_jobs=config.n_jobs)

        landmarks_dist = witness_complex_train.landmarks_dist
        landmark_dist_train[:, :, batch_i] = landmarks_dist

    for batch_i, (X_batch, label_barch) in enumerate(dataloader_eval):
        witness_complex_train = WitnessComplex(landmarks=X_batch, witnesses=X_witnesses,
                                               n_jobs=config.n_jobs)
        witness_complex_train.compute_metric_optimized(n_jobs=config.n_jobs)

        landmarks_dist = witness_complex_train.landmarks_dist
        landmark_dist_eval[:, :, batch_i] = landmarks_dist

    # save torch files
    torch.save(dataloader_train, os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DATALOADER_TRAIN)))
    torch.save(dataloader_eval, os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DATALOADER_EVAL)))
    torch.save(landmark_dist_train,
               os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DISTANCES_TRAIN)))
    torch.save(landmark_dist_eval,
               os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DISTANCES_EVAL)))

    # make entry in global register
    df = pd.DataFrame.from_dict(config.create_dict()).reset_index()
    columns = ['uid', 'dataset', 'batch_size', 'seed', 'root_path']
    df = df[columns]
    df.root_path = os.path.join(config.root_path, config.uid)
    df.to_csv(config.global_register, mode='a', header=False)


def fetch_data(uid: str = None, path_global_register: str = None, path_to_data: str = None) -> Tuple[
    DataLoader, torch.Tensor, DataLoader, torch.Tensor]:
    '''
    Get dataloader and landmark distances for offline computed witness complexes.
    Either through uid and global register path or directly located in a root folder.
    :param uid: Unique identifier of data to load
    :param path_global_register: path to global register
    :param path_to_data: path to data to load
    :return:
    '''

    assert (uid is not None and path_global_register is not None) or (path_to_data is not None)

    if (uid is not None and path_global_register is not None):
        fetch_from_register = True
    else:
        fetch_from_register = False

    if fetch_from_register and (path_to_data is not None):
        print('WARNING: path_to_data is ignored')

    # get path from register OR use provided path
    if fetch_from_register:
        df_register = pd.read_csv(path_global_register)
        path_to_data = df_register[df_register['uid'] == uid].root_path.values[0]
    else:
        pass

    # fetch and return data.
    dataloader_train = torch.load(os.path.join(path_to_data,'{}.pt'.format(NAME_DATALOADER_TRAIN)))
    dataloader_eval = torch.load(os.path.join(path_to_data,'{}.pt'.format(NAME_DATALOADER_EVAL)))
    landmark_dist_train = torch.load(os.path.join(path_to_data,'{}.pt'.format(NAME_DISTANCES_TRAIN)))
    landmark_dist_eval = torch.load(os.path.join(path_to_data,'{}.pt'.format(NAME_DISTANCES_EVAL)))

    return dataloader_train, landmark_dist_train, dataloader_eval, landmark_dist_eval