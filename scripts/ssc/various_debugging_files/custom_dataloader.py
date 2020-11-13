import torch
from torch.utils.data import TensorDataset, DataLoader

labels = torch.Tensor(range(100))


dataset = TensorDataset(labels)
dataloader = DataLoader(dataset, batch_size=20, pin_memory=True, drop_last=True,shuffle=False)

for i, labels in enumerate(dataloader):
    print(labels)


for i, labels in enumerate(dataloader):
    if i == 0:
        print(labels)