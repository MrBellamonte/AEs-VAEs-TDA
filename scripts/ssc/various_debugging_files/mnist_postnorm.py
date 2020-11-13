import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader



root_path = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist_old'
root_path_new = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WitnessComplexes/mnist'
dataloader_names = ['dataloader_train.pt','dataloader_test.pt','dataloader_eval.pt']
wcs = ['MNIST_offline-bs128-seed838-noiseNone-4f608157','MNIST_offline-bs256-seed838-noiseNone-4a5487de','MNIST_offline-bs512-seed838-noiseNone-ced06774','MNIST_offline-bs1024-seed838-noiseNone-6f31dea2']
#wcs = ['MNIST_offline-bs64-seed838-noiseNone-20738678']




for wc_1 in wcs:
    bs = int(wc_1.split('-')[1][2:])
    for dataloader_name in dataloader_names:

        complete_path = os.path.join(root_path,wc_1,dataloader_name)

        dataloader = torch.load(complete_path)




        data_complete = np.array(())
        label_complete = np.array(())

        for bs_i, (data,label) in enumerate(dataloader):
            if bs_i%20==0:
                print('{} out of {}'.format(bs_i, len(dataloader)))
            if bs_i == 0:
                data_complete = data.numpy()
            else:
                data_complete = np.vstack((data_complete, data.numpy()))
            label_complete = np.append(label_complete, label.numpy())

        data_complete_tensor = torch.from_numpy(data_complete)
        label_complete_tensor = torch.from_numpy(label_complete)

        data_complete_tensor = data_complete_tensor/255

        dataset = TensorDataset(data_complete_tensor,label_complete_tensor)

        dataloader_new = DataLoader(dataset, batch_size=bs,shuffle = False, drop_last=True)


        mistakes = 0

        for bs_0, (datanew,labelnew) in enumerate(dataloader_new):
            for bs_1, (dataold, labelold) in enumerate(dataloader):
                if bs_0 == bs_1:
                    #print('data equal: {} ---- lables equal: {}'.format(torch.all(torch.eq(dataold/255, datanew)),torch.all(torch.eq(labelold, labelnew))))
                    if (torch.all(torch.eq(dataold/255, datanew)) and torch.all(torch.eq(labelold, labelnew))):
                        pass
                    else:
                        mistakes += 1
                elif bs_1 > bs_0:
                    break
                else:
                    pass
        print(wc_1)
        print(dataloader_name)
        print('SAME LENGTH:  {} - MISTAKES: {}'.format((len(dataloader) == len(dataloader_new)),mistakes))


        torch.save(dataloader_new,
                   os.path.join(root_path_new,wc_1,dataloader_name))
        if (len(dataloader) == len(dataloader_new)) and mistakes == 0:
            print('SUCCESS')
