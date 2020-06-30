"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
import inspect
import os
import pickle
import uuid

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from collections import defaultdict

from src.model.autoencoders import autoencoder

from torchph.pershom import pershom_backend
vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1

# config
DEVICE  = "cuda"

#todo: get rid of this mapping, not a nice solution.
model_mapping = {
    'autoencoder' : autoencoder
}




def l1_loss(x_hat, x, reduce=True):
    """
    L1 loss used for reconstruction.
    """
    l = (x - x_hat).abs().view(x.size(0), - 1).sum(dim=1)
    if reduce:
        l = l.mean()
    return l



def check_config(config):
    #todo: Think about if it makes sense to create a "config" class....
    #todo: Should contain information on dataset for uuid


    assert 'train_args' in config
    train_args = config['train_args']

    assert 'learning_rate' in train_args
    assert 0 < train_args['learning_rate']

    assert 'batch_size' in train_args
    assert 0 < train_args['batch_size']

    assert 'n_epochs' in train_args
    assert 0 < train_args['n_epochs']

    assert 'rec_loss_w' in train_args
    assert 'top_loss_w' in train_args

    # check model-speficic args
    assert 'model_args' in config
    model_args = config['model_args']
    assert 'class_id' in model_args
    assert model_args['class_id'] in model_mapping
    assert 'kwargs' in model_args
    kwargs = model_args['kwargs']
    s = inspect.getfullargspec(model_mapping[model_args['class_id']].__init__)
    for a in s.kwonlyargs:
        assert a in kwargs
    try:
        create_uuid(config)
    except:
        print("Failed to create unique ID")


def create_uuid(config):
    uuid_suffix = str(uuid.uuid4())[:8]

    uuid_str = '{}-{}-lr{}-bs{}-nep{}-rlw{}-tlw{}'.format(config['model_args']['class_id'],
                                                          '-'.join(str(x) for x in
                                                                   config['model_args']['kwargs'][
                                                                       'size_hidden_layers'])+'-'+str(
                                                              config['model_args']['kwargs'][
                                                                  'latent_dim']),
                                                          int(1000*config['train_args']['learning_rate']),
                                                          config['train_args']['batch_size'],
                                                          config['train_args']['n_epochs'],
                                                          int(100*config['train_args']['rec_loss_w']),
                                                          int(100*config['train_args']['top_loss_w']))

    return uuid_str+'-'+uuid_suffix


def train(data, config, root_folder):

    # HARD-CODED conifg
    ball_radius = 1.0 #only affects the scaling


    check_config(config)

    train_args = config['train_args']
    model_args = config['model_args']

    model_class = model_mapping[model_args['class_id']]

    model = model_class(**model_args['kwargs']).to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=train_args['learning_rate'])

    dl = DataLoader(data,
                    batch_size=train_args['batch_size'],
                    shuffle=True,
                    drop_last=True)

    log = defaultdict(list)

    model.train()

    for epoch in range(1,train_args['n_epochs']+1):

        for x in dl:
            x = x.to(DEVICE)

            # Get reconstruction x_hat and latent
            # space representation z
            x_hat, z = model(x.float())

            # Set both losses to 0 in case we ever want to
            # disable one and still use the same logging code.
            top_loss = torch.tensor([0]).type_as(x_hat)

            # Computes l1-reconstruction loss
            rec_loss = l1_loss(x_hat, x, reduce=True)

            # For each branch in the latent space representation,
            # we enforce the topology loss and track the lifetimes
            # for further analysis.
            lifetimes = []
            pers = vr_l1_persistence(z[:,:].contiguous(), 0, 0)[0][0]

            if pers.dim() == 2:
                pers = pers[:, 1]
                lifetimes.append(pers.tolist())
                top_loss += (pers-2.0*ball_radius).abs().sum()

            # Log lifetimes as well as all losses we compute
            log['lifetimes'].append(lifetimes)
            log['top_loss'].append(top_loss.item())
            log['rec_loss'].append(rec_loss.item())

            loss = train_args['rec_loss_w']*rec_loss + train_args['top_loss_w']*top_loss # HARD-CODED: equal weight

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
            epoch,
            np.array(log['rec_loss'][-int(len(data)/train_args['batch_size']):]).mean()*train_args['rec_loss_w'],
            np.array(log['top_loss'][-int(len(data)/train_args['batch_size']):]).mean()*train_args['top_loss_w']))

    # Create a unique base filename
    the_uuid = create_uuid(config)

    path = os.path.join(root_folder, the_uuid)
    os.makedirs(path)
    config['uuid'] = the_uuid

    # Save model
    torch.save(model.state_dict(), '.'.join([path + '/model', 'pht']))


    # Save the config used for training as well as all logging results
    out_data = [config, log]
    file_ext = ['config', 'log']
    for x, y in zip(out_data, file_ext):
        with open('.'.join([path + '/'+ y, 'pickle']), 'wb') as fid:
            pickle.dump(x, fid)
