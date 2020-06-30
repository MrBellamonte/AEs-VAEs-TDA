import torch
import numpy as np
import matplotlib.pyplot as plt
from torchph.pershom import vr_persistence_l1

device = "cuda"

np.random.seed(1234)
toy_data = np.random.rand(300, 2)

plt.figure()
plt.plot(toy_data[:, 0], toy_data[:, 1], 'b.', markersize=3)
plt.title('Toy data');


X = torch.tensor(
    toy_data,
    device=device,
    requires_grad=True)

opt = torch.optim.Adam([X], lr=0.01)

for i in range(1,100+1):
    pers = vr_persistence_l1(X, 1, 0)
    h_0 = pers[0][0]

    lt = h_0[:, 1] # H0 lifetimes
    loss = (lt - 0.1).abs().sum()

    if i % 20 == 0 or i == 1:
        print('Iteration: {:3d} | Loss: {:.2f}'.format(i, loss.item()))

    opt.zero_grad()
    loss.backward()
    opt.step()

X = X.cpu().detach().numpy()
plt.figure()
plt.plot(X[:, 0], X[:, 1], 'b.');