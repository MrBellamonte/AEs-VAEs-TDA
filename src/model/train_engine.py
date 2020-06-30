"""train_engine.py
source: https://github.com/c-hofer/COREL_icml2019

modified version, tailored to our needs
"""
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


def l1_loss(x_hat, x, reduce=True):
    """
    L1 loss used for reconstruction.
    """
    l = (x - x_hat).abs().view(x.size(0), - 1).sum(dim=1)
    if reduce:
        l = l.mean()
    return l


def train(data):

    # HARD-CODED conifg
    #todo: get rid of it!
    batch_size = 32
    rec_loss_w = 1
    top_loss_w = 1
    ball_radius = 1.0 #only affects the scaling
    epoch = 10
    #####################

    model = autoencoder().to(DEVICE)
    optimizer = Adam(
        model.parameters(),
        lr=1e-3)

    dl = DataLoader(data,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True)

    log = defaultdict(list)

    model.train()

    for epoch in range(1,epoch+1):

        for x in dl:
            x = x.to(DEVICE)

            # Get reconstruction x_hat and latent
            # space representation z
            x_hat, z = model(x.float())

            # Set both losses to 0 in case we ever want to
            # disable one and still use the same logging code.
            top_loss = torch.tensor([0]).type_as(x_hat)
            rec_loss = torch.tensor([0]).type_as(x_hat)

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

            loss = 1*rec_loss + 1*top_loss # HARD-CODED: equal weight

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
            epoch,
            np.array(log['rec_loss'][-int(len(data)/batch_size):]).mean()*rec_loss_w,
            np.array(log['top_loss'][-int(len(data)/batch_size):]).mean()*top_loss_w))

