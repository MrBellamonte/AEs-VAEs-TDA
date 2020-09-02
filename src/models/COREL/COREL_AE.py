import torch
from torch import nn

from src.models.autoencoder.base import AutoencoderModel

from torchph.pershom import pershom_backend

vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1

# hard coded
BALL_RADIUS = 1

class COREL_H0_Autoencoder(nn.Module):


    def __init__(self, autoencoder, rec_loss_func, top_loss_func,lam_t=1.,lam_r=1.):
        super().__init__()
        self.lam_t = lam_t
        self.lam_r = lam_r
        self.autoencoder = autoencoder
        self.rec_loss_func = rec_loss_func
        self.top_loss_func = top_loss_func

    def forward(self, x):

        if x.is_cuda:
            pass
        else:
            print('HERE')
            x = x.to('cuda')

        latent = self.autoencoder.encode(x)
        x_hat = self.autoencoder.decode(latent)



        rec_loss = self.rec_loss_func(x_hat, x)

        top_loss = torch.tensor([0]).type_as(x_hat)
        lifetimes = []
        pers = vr_l1_persistence(latent[:, :].contiguous(), 0, 0)[0][0]

        if pers.dim() == 2:
            pers = pers[:, 1]
            lifetimes.append(pers.tolist())
            top_loss_func = self.top_loss_func
            top_loss += top_loss_func.forward(pers, 2.0*BALL_RADIUS*torch.ones_like(pers))

        loss = self.lam_r*rec_loss+self.lam_t*top_loss
        loss_components = {
            'loss.autoencoder': rec_loss,
            'loss.topo_error': top_loss
        }

        return (
            loss,
            loss_components
        )



    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)