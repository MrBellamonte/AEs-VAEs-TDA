import numpy as np

from sklearn.preprocessing import normalize

from src.datasets.datasets import Spheres
from src.evaluation.eval import Multi_Evaluation
from src.evaluation.measures_optimized import MeasureCalculator
from src.utils.dict_utils import avg_array_in_dict

if __name__ == "__main__":
    dataset = Spheres()
    try:
        dataset.sample_manifold()
    except AttributeError as err:
        print(err)
        print('Manifold not evaluated!')
    # X = np.random.random((50, 2))
    # Y = np.random.random((50, 1))
    # Z = X.copy()/10
    #
    # X_norm = normalize(X, axis=0)
    # Z_norm = normalize(Z, axis=0)
    #
    # metric_calc = MeasureCalculator(X = X, Z = Z, k_max=15)
    # metric_calc_norm = MeasureCalculator(X=X_norm, Z=Z_norm, k_max=15)
    #
    #
    # ks = [5,10,15]
    # evaluator = Multi_Evaluation(
    #     dataloader=None, seed=1, model=None)
    # ev_result = evaluator.get_multi_evals(
    #     X, X_norm, Y, ks=ks)
    # result_avg = avg_array_in_dict(ev_result)
    # alpha = 1

