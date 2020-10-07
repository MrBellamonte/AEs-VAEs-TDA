from scripts.ssc.Competitors.config_library.tsne import swissroll_test as swissroll_test_tsne
from scripts.ssc.Competitors.config_library.umap import swissroll_test as swissroll_test_umap
from src.competitors.train_engine import simulator_competitor

if __name__ == "__main__":
    simulator_competitor(swissroll_test_umap)


