import inspect
import itertools
import uuid
from dataclasses import dataclass
from typing import Type, List

from src.competitors.competitor_models import Competitor, tSNE
from src.datasets.datasets import DataSet, SwissRoll
from src.evaluation.config import ConfigEval
from src.models.loss_collection import Loss
from src.utils.config_utils import (
    get_keychain_value, get_kwargs,
    dictionary_to_string)


@dataclass
class Config_Competitors:
    __slots__ = ['model_class',
                 'model_kwargs',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid',
                 'seed']
    model_class: Type[Competitor]
    model_kwargs: dict
    dataset: Type[DataSet]
    sampling_kwargs: dict
    eval: ConfigEval
    uid: str
    seed: int


    def __post_init__(self):
        self.uid = self.creat_uuid()
        self.check()

    def creat_uuid(self):
        if self.uid == '':
            unique_id = str(uuid.uuid4())[:8]

            uuid_model = '{model}-{model_kwargs}-seed{seed}'.format(
                model=self.model_class.__name__,
                model_kwargs=dictionary_to_string(self.model_kwargs),
                seed = str(self.seed))

            if 'root_path' in self.sampling_kwargs:
                sampling_kwargs2 = self.sampling_kwargs.copy()
                sampling_kwargs2.pop('root_path')
            else:
                sampling_kwargs2 = self.sampling_kwargs.copy()

            uuid_data = '{dataset}{object_kwargs}{sampling_kwargs}-'.format(
                dataset=self.dataset.__class__.__name__,
                object_kwargs=get_kwargs(self.dataset),
                sampling_kwargs=dictionary_to_string(sampling_kwargs2),

            )
            return uuid_data+uuid_model+'-'+ unique_id
        else:
            return self.uid

    def create_dict(self):
        ret_dict = dict()
        for slot in self.__slots__:

            if isinstance(getattr(self, slot), (Loss, DataSet)):
                ret_dict.update({slot: getattr(self, slot).__class__.__name__})
            else:
                ret_dict.update({slot: getattr(self, slot)})

        # need to manually change model class
        ret_dict.update(dict(model_class=self.model_class.__name__))

        return ret_dict

    def check(self):
        s = inspect.getfullargspec(self.model_class.__init__)
        for a in s.kwonlyargs:
            assert a in self.model_kwargs
        assert self.creat_uuid()
        assert self.create_dict()


    def create_id_dict(self):
        return dict(
            uid = self.uid,
            model = self.model_class.__name__,
        )


@dataclass
class ConfigGrid_Competitors:
    __slots__ = ['model_class',
                 'model_kwargs',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid',
                 'experiment_dir',
                 'seed',
                 'verbose'
                 ]
    model_class: List[Type[Competitor]]
    model_kwargs: List[dict]
    dataset: List[Type[DataSet]]
    sampling_kwargs: List[dict]
    eval: List[ConfigEval]
    uid: List[str]
    experiment_dir: str
    seed: int
    verbose: str


    def configs_from_grid(self):

        grid = dict()

        for slot in (set(self.__slots__)-set(['experiment_dir','seed', 'verbose'])):
            grid.update({slot: getattr(self, slot)})
        tmp = list(get_keychain_value(grid))
        values = [x[1] for x in tmp]
        key_chains = [x[0] for x in tmp]

        ret = []

        for v in itertools.product(*values):
            ret_i = {'seed': self.seed}

            for kc, kc_v in zip(key_chains, v):
                tmp = ret_i
                for k in kc[:-1]:
                    if k not in tmp:
                        tmp[k] = {}

                    tmp = tmp[k]

                tmp[kc[-1]] = kc_v

            ret.append(Config_Competitors(**ret_i))

        return ret



placeholder_config_competitors = Config_Competitors(
    model_class = tSNE,
    model_kwargs=dict(),
    dataset=SwissRoll(),
    sampling_kwargs={
        'n_samples': [2560]
    },
    eval=[ConfigEval(
        active = True,
        evaluate_on = None,
        save_eval_latent = True,
        save_train_latent = True,
        online_visualization = False,
        k_min=5,
        k_max=20,
        k_step=5,
    )],
    uid = ['uid'],
    seed = 123123,
)