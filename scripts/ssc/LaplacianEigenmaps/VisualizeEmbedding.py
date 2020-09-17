import time

from sklearn.manifold import SpectralEmbedding

from src.datasets.datasets import SwissRoll, Spheres
from src.utils.plots import plot_classes_qual

if __name__ == "__main__":


    dataset = SwissRoll()

    data, color = dataset.sample(n_samples=2560)

    start = time.time()
    embedding = SpectralEmbedding(n_components=2,n_jobs=1, n_neighbors=90)

    X_transformed = embedding.fit_transform(data)
    end = time.time()
    print('It took: {}'.format(end - start))

    plot_classes_qual(data = X_transformed, labels=color, path_to_save= None, title = None, show = True)