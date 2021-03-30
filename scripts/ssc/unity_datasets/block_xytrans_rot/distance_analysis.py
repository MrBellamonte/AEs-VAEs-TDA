import os

import torch

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/src/datasets/simulated/xy_trans_rot'

    dataset = torch.load(os.path.join(path, 'full_dataset.pt'))
    posititon = torch.load(os.path.join(path, 'position.pt'))
    images = torch.load(os.path.join(path, 'images.pt'))

    ind0 = torch.where((posititon == torch.Tensor([0, 0, 0])).all(dim=1))[0]

    ind_xp = torch.where((posititon == torch.Tensor([0.5, 0, 0])).all(dim=1))[0]
    ind_xm = torch.where((posititon == torch.Tensor([-0.5, 0, 0])).all(dim=1))[0]

    ind_yp = torch.where((posititon == torch.Tensor([0, 0.5, 0])).all(dim=1))[0]
    ind_ym = torch.where((posititon == torch.Tensor([0, -0.5, 0])).all(dim=1))[0]

    ind_90 = torch.where((posititon == torch.Tensor([0, 0, 90])).all(dim=1))[0]
    ind_180 = torch.where((posititon == torch.Tensor([0, 0, 180])).all(dim=1))[0]
    ind_270 = torch.where((posititon == torch.Tensor([0, 0, 270])).all(dim=1))[0]

    dist_xp = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_xp,:,:,:].view(1, 3*480*320))
    dist_xm = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_xm,:,:,:].view(1, 3*480*320))
    dist_yp = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_yp,:,:,:].view(1, 3*480*320))
    dist_ym = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_ym,:,:,:].view(1, 3*480*320))
    dist_90 = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_90,:,:,:].view(1, 3*480*320))
    dist_180 = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_180,:,:,:].view(1, 3*480*320))
    dist_270 = torch.cdist(images[ind0,:,:,:].view(1, 3*480*320),images[ind_270,:,:,:].view(1, 3*480*320))

    print([float(dist_xp), float(dist_xm), float(dist_yp), float(dist_ym), float(dist_90), float(dist_180), float(dist_270)])


