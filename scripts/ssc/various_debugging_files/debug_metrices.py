import numpy as np

from sklearn.preprocessing import normalize

from src.evaluation.measures_optimized import MeasureCalculator

if __name__ == "__main__":



    X = np.random.random((50,2))
    Y = np.random.random((50, 1))
    Z = X.copy()/10

    X_norm = normalize(X, axis=0)
    Z_norm = normalize(Z, axis=0)

    metric_calc = MeasureCalculator(X = X, Z = Z, k_max=15)
    metric_calc = MeasureCalculator(X=X_norm, Z=Z_norm, k_max=15)
    alpha = 1

    print(alpha)

