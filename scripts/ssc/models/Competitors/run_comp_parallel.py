from joblib import Parallel, delayed

from scripts.ssc.models.Competitors import swissroll_umap_grid
from src.competitors.train_engine import simulator_competitor

if __name__ == "__main__":
    Parallel(n_jobs=6)(delayed(simulator_competitor)(config) for config in swissroll_umap_grid.con)

