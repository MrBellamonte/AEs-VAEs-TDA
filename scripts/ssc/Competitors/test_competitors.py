from scripts.ssc.Competitors.config_library.tsne import swissroll_test as swissroll_test_tsne
from scripts.ssc.Competitors.config_library.umap import (
    swissroll_test as swissroll_test_umap,
    swissroll_umap_gridcccc)
from src.competitors.train_engine import simulator_competitor

if __name__ == "__main__":
    #simulator_competitor(swissroll_test_umap)

    grid = swissroll_umap_gridcccc.configs_from_grid()

    print(len(grid))

