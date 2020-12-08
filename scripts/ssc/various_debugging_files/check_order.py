import os

import torch

if __name__ == "__main__":

    root_to_data = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs128-seed838-noiseNone-4f608157'

    root_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/mnist_umap_nncomp/test'
    # load data batchwise into df, create integer id + batch id
    dataloader = torch.load(os.path.join(root_to_data,'dataloader_train.pt'))

    Xs = []
    ys = []
    for i, (X_batch, label_batch) in enumerate(dataloader):
        if i == 0:
            X0 = X_batch
        Xs.append(X_batch)
        ys.append(label_batch)

    X_new = torch.cat(Xs,dim = 0)
    y_new = torch.cat(ys, dim=0)

    print(y_new.shape)
    print(X_new.shape)

    print(X_new[0:128,:].shape)

    print(torch.all(X0.eq(X_new[0:128,:])))

