from dataclasses import dataclass


@dataclass
class ConfigEval:
    active: bool
    evaluate_on: str
    save_eval_latent: bool
    save_train_latent: bool
    online_visualization: bool
    k_min: int
    k_max: int
    k_step: int
    eval_manifold: bool = False

    def __post_init__(self):
        self.check()

    def ks(self):
        return list(range(self.k_min, self.k_max + self.k_step, self.k_step))

    def check(self):
        assert self.evaluate_on in ['validation', 'test', None]

        assert self.k_min > 0
        assert self.k_max > 0
        assert self.k_step > 0