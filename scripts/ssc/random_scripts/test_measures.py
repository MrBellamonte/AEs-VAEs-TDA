from src.datasets.datasets import SwissRoll
from src.evaluation.eval import Multi_Evaluation

dataset_sampler = SwissRoll()
data, label = dataset_sampler.sample(1000, seed=1)
data2, label = dataset_sampler.sample(1000, seed=2)



evaluator = Multi_Evaluation(seed=1)
ev_result = evaluator.get_multi_evals(
    data, data, label, ks=[5,10,15])


print(ev_result)