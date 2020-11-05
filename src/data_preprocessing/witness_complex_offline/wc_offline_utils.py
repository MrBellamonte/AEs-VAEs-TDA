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


def compute_wc(dataloader,  X_witnesses, config):
    landmark_dist= torch.zeros(len(dataloader),config.batch_size, config.batch_size)

    # save dataloader, landmarks distances for eval and train
    for batch_i, (X_batch, label_batch) in enumerate(dataloader):
        witness_complex = WitnessComplex(landmarks=X_batch, witnesses=X_witnesses,
                                               n_jobs=config.n_jobs)
        witness_complex.compute_metric_optimized(n_jobs=config.n_jobs)

        landmarks_dist_batch = witness_complex.landmarks_dist
        landmark_dist[batch_i,:, :] = landmarks_dist_batch
    return landmark_dist


def compute_kNN(dataloader,  X_witnesses, config):
    landmark_dist= torch.zeros(len(dataloader),config.batch_size, config.batch_size)

    # save dataloader, landmarks distances for eval and train
    for batch_i, (X_batch, label_batch) in enumerate(dataloader):
        witness_complex = WitnessComplex(landmarks=X_batch, witnesses=X_witnesses,
                                               n_jobs=config.n_jobs)
        witness_complex.compute_metric_optimized(n_jobs=config.n_jobs)

        landmarks_dist_batch = witness_complex.landmarks_dist
        landmark_dist[batch_i,:, :] = landmarks_dist_batch
    return landmark_dist


def compute_wc_offline(config: ConfigWC):
    '''
    Utility to compute distances between landmarks "offline" and save dataloader/distance matrices for later training.
    '''

    torch.manual_seed(config.seed)

    # get data, i.e. sample or load
    dataset = config.dataset
    X_train, y_train = dataset.sample(**config.sampling_kwargs, seed=config.seed, train=True)

    X_test, y_test = dataset.sample(**config.sampling_kwargs, seed=config.seed, train=False)

    # "assign witnesses" -> in the future maybe also through sub-sampling
    X_witnesses_train = select_witnesses(X_train, **config.wc_kwargs)
    X_witnesses_test = select_witnesses(X_test, **config.wc_kwargs)

    # split datasets into train and eval
    dataset_tensor_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    eval_size = int(config.eval_size*y_train.size)
    train_size = int(y_train.size-eval_size)

    train_dataset, validation_dataset = random_split(dataset_tensor_train, (train_size, eval_size))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # load batches
    dataloader_train = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, shuffle=False
    )
    dataloader_eval = DataLoader(
        validation_dataset, batch_size=config.batch_size, pin_memory=True, drop_last=True,
        shuffle=False
    )
    dataloader_test= DataLoader(
        test_dataset, batch_size=config.batch_size, pin_memory=True, drop_last=True,
        shuffle=False
    )

    # compute witness complexes
    if config.verbose:
        print('Comopute witness complex for training data...')
    landmark_dist_train = compute_wc(dataloader_train,X_witnesses_train,config)
    if config.verbose:
        print('Comopute witness complex for evaluation data...')
    landmark_dist_eval = compute_wc(dataloader_eval,X_witnesses_train,config)
    if config.verbose:
        print('Comopute witness complex for test data...')
    landmark_dist_test = compute_wc(dataloader_test,X_witnesses_test,config)


    if config.verbose:
        print('Save data...')
    # save dataloader
    torch.save(dataloader_train, os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DATALOADER_TRAIN)))
    torch.save(dataloader_eval, os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DATALOADER_EVAL)))
    torch.save(dataloader_test, os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DATALOADER_TEST)))

    # save landmark distance matrics
    torch.save(landmark_dist_train,
               os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DISTANCES_TRAIN)))
    torch.save(landmark_dist_eval,
               os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DISTANCES_EVAL)))
    torch.save(landmark_dist_test,
               os.path.join(config.root_path, config.uid, '{}.pt'.format(NAME_DISTANCES_TEST)))

    # make entry in global register
    df = pd.DataFrame.from_dict(config.create_dict()).reset_index()
    columns = ['uid', 'dataset', 'batch_size', 'seed', 'root_path']
    df = df[columns]
    df.root_path = os.path.join(config.root_path, config.uid)
    df.to_csv(config.global_register, mode='a', header=False)


def fetch_data(uid: str = None, path_global_register: str = None, path_to_data: str = None, type: str = 'train') -> Tuple[
    DataLoader, torch.Tensor, DataLoader, torch.Tensor]:
    '''
    Get dataloader and landmark distances for offline computed witness complexes.
    Either through uid and global register path or directly located in a root folder.
    :param uid: Unique identifier of data to load
    :param path_global_register: path to global register
    :param path_to_data: path to data to load
    :param type: define which data to loade ('train', 'eval', 'test')
    :return:
    '''

    assert type in ['train', 'eval', 'test','train' ,'validation', 'testing']
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
    if (type == 'train') or  (type == 'training'):
        dl_name = NAME_DATALOADER_TRAIN
        dm_name = NAME_DISTANCES_TRAIN
        dmx_name = NAME_DISTANCES_X_TRAIN
    elif (type == 'eval') or (type == 'validation'):
        dl_name = NAME_DATALOADER_EVAL
        dm_name = NAME_DISTANCES_EVAL
        dmx_name = NAME_DISTANCES_X_EVAL
    else:
        dl_name = NAME_DATALOADER_TEST
        dm_name = NAME_DISTANCES_TEST
        dmx_name = NAME_DISTANCES_X_TEST

    dataloader = torch.load(os.path.join(path_to_data,'{}.pt'.format(dl_name)))
    landmark_distances =  torch.load(os.path.join(path_to_data,'{}.pt'.format(dm_name)))

    if os.path.exists(os.path.join(path_to_data,'{}.pt'.format(dmx_name))):
        data_distances =  torch.load(os.path.join(path_to_data,'{}.pt'.format(dmx_name)))
    else:
        data_distances = False

    return dataloader, landmark_distances, data_distances


def get_kNNmask(landmark_distances, num_batches, batch_size, k):
    pair_mask_all = torch.ones((num_batches, batch_size, batch_size))
    for batch_i in range(num_batches):
        landmark_distances_batch = landmark_distances[batch_i,:,:]
        sorted, indices = torch.sort(landmark_distances_batch)
        kNN_mask = torch.zeros((batch_size, batch_size), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)
        pair_mask_all[batch_i, :, :] = kNN_mask

    return pair_mask_all