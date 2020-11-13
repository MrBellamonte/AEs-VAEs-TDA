from scripts.ssc.evaluation.final_evaluators.offline_evaluation import offline_eval_WAE
from src.evaluation.config import ConfigEval

exp_dir = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WAE/openai/rotating'
stw = 'Unity'
evalconfig = ConfigEval(
        active=True,
        evaluate_on='test',
        eval_manifold=False,
        save_eval_latent=True,
        save_train_latent=True,
        online_visualization=False,
        quant_eval=False,
        k_min=2,
        k_max=10,
        k_step=2)
offline_eval_WAE(exp_dir, evalconfig, stw)