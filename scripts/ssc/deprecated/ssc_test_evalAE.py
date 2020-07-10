from torch import Tensor
from torch.utils.data import TensorDataset

from src.datasets.shapes import dsphere
from src.model.COREL.eval_engine import get_model, get_latentspace_representation

if __name__ == "__main__":

    path = '/home/simonberg/PycharmProjects/MT-VAEs-TDA/output/2020-06-30/autoencoder-128-64-32-2-lr1-bs512-nep5-rlw100-tlw100-9120e14f/'

    X, y = dsphere(n=1024*4, r=10, d=3)

    dataset = TensorDataset(Tensor(X), Tensor(y))

    model = get_model(path)

    X,Y,Z = get_latentspace_representation(model,dataset)