import os

import torch
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances


def calc_matching_fraction(pairskNN_,pairskNNWC_, k= None, verbose = True):
    pairskNN = (pairskNN_.bool() + pairskNN_.T.bool()).float()
    pairskNNWC = (pairskNNWC_.bool()+pairskNNWC_.T.bool()).float()


    npairs_missed = float(((pairskNNWC-pairskNN) == 1).sum())
    npairs_added = float(((pairskNN-pairskNNWC) == 1).sum())

    paris_matched =  ((pairskNNWC == 1.0)*(pairskNN == 1.0)).float().sum()

    if verbose:
        print('----- K = {} -----'.format(k))

        print('Pairs in kNNWC: {}'.format(float(pairskNNWC.sum())/2))
        print('Pairs in kNN: {}'.format(float(pairskNN.sum())/2))

        print('Pairs matched: {}'.format(float(npairs_missed)/2))
        print('Pairs in kNN and not in kNNWC: {}'.format(float(npairs_added)/2))
        print('Pairs in kNNWC and not in kNN: {}'.format(float(npairs_missed)/2))

    return float(npairs_missed)/2,float(npairs_added)/2,float(paris_matched)/2



if __name__ == "__main__":
    k = 1
    dir_path64 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs64-seed838-noiseNone-20738678'
    dir_path128 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs128-seed838-noiseNone-4f608157'
    dir_path256 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs256-seed838-noiseNone-4a5487de'
    dir_path512 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs512-seed838-noiseNone-ced06774'
    dir_path1024 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist/MNIST_offline-bs1024-seed838-noiseNone-6f31dea2'


    dir_path = dir_path64
    euc_dist = torch.load(os.path.join(dir_path, 'dist_X_all_train.pt'))
    landmark_dist = torch.load(os.path.join(dir_path, 'landmark_dist_train.pt'))
    dataloader = torch.load(os.path.join(dir_path, 'dataloader_train.pt'))

    print(euc_dist.shape)
    print(landmark_dist.shape)

    npairs_missed_list, npairs_added_list, paris_matched_list = [],[],[]

    #for i in range(euc_dist.shape[0]):
    for i, (data, labels) in enumerate(dataloader):

        pw_labels = pairwise_distances(labels.numpy().reshape(labels.shape[0], -1),labels.numpy().reshape(labels.shape[0], -1))
        pw_labels[pw_labels!=0] = 1

        sns.heatmap(pw_labels, cmap='coolwarm', robust=True)
        plt.show()

        euc_dist_bi = euc_dist[i,:,:]

        sns.heatmap((euc_dist_bi.numpy()/euc_dist_bi.max().numpy()), cmap='coolwarm', robust=True)
        plt.show()

        sorted, indices = torch.sort(euc_dist_bi)
        pairskNN = torch.zeros((euc_dist.shape[1], euc_dist.shape[1]), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)


        landmark_dist_bi = landmark_dist[i, :, :]
        sorted, indices = torch.sort(landmark_dist_bi)
        pairskNNWC = torch.zeros((euc_dist.shape[1], euc_dist.shape[1]), device='cpu').scatter(1, indices[:, 1:(k+1)], 1)

        sns.heatmap((landmark_dist_bi.numpy()/landmark_dist_bi.max().numpy()), cmap='coolwarm', robust=True)
        plt.show()

        npairs_missed_i,npairs_added_i,paris_matched_i = calc_matching_fraction(pairskNN,pairskNNWC,k)
        npairs_missed_list.append(npairs_missed_i)
        npairs_added_list.append(npairs_added_i)
        paris_matched_list.append(paris_matched_i)


    print('FINAL')
    print('Pairs matched: {}'.format(sum(paris_matched_list)/len(paris_matched_list)))
    print('Pairs in kNN and not in kNNWC: {}'.format(sum(npairs_added_list)/len(npairs_added_list)))
    print('Pairs in kNNWC and not in kNN: {}'.format(sum(paris_matched_list)/len(paris_matched_list)))
