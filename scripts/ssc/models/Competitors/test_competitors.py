from scripts.ssc.models.Competitors.config_libraries.umap_mnist import (
    umap_mnist_test_local,
    umap_mnist_euler_1)
from src.competitors.train_engine import simulator_competitor

if __name__ == "__main__":
    for config in umap_mnist_test_local.configs_from_grid():
        print(config)


