import torch
import numpy as np

from sklearn.preprocessing import normalize

from src.evaluation.measures_optimized import MeasureCalculator

if __name__ == "__main__":



    X = np.array(((0,1,1,0),(0,0,1,1),(1,1,0,0),(1,1,0,0)))
    X = np.array(((0, 0, 1, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 0, 0)))
    X_ = torch.from_numpy(X)
    X_tot = (X_.bool() +  X_.bool().t()).int()
    print(X_tot)

    Z = np.array(((0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1), (1, 1, 0, 0)))
    Z = np.array(((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 0, 0)))
    Z_ = torch.from_numpy(Z)
    Z_tot = (Z_.bool()+Z_.bool().t()).int()
    print(Z_tot)

    tot_pairings = int(X_tot.sum())
    missed_pairings = int(((X_tot-Z_tot) == 1).sum())
    print(tot_pairings)
    print(missed_pairings)

    print('fraction: {}'.format((tot_pairings-missed_pairings)/tot_pairings))