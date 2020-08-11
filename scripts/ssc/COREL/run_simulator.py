import datetime
import os

from src.models.COREL.train_engine import simulator_COREL, simulator_COREL_2

from scripts.ssc.COREL.config_library import grid_spheres_ts, grid_spheres, grid_spheres_ts_sq, grid_spheres_ts_large

if __name__ == "__main__":
    #simulator_COREL_2(grid_spheres)
    #simulator_COREL_2(grid_spheres_ts)
    #simulator_COREL_2(grid_spheres_ts_sq)
    simulator_COREL_2(grid_spheres_ts_large)
