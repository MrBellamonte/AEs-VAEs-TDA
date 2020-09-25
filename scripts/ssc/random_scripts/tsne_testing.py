from sklearn.manifold import TSNE

from src.datasets.datasets import SwissRoll

if __name__ == "__main__":
    dataset = SwissRoll()

    data, labels = dataset.sample(n_samples=100)

    model = TSNE()



