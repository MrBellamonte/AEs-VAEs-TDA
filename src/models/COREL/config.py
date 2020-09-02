import inspect
import itertools
import uuid
from dataclasses import dataclass
from typing import Type, List

import torch

from src.datasets.datasets import DataSet
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import Autoencoder_MLP
from src.models.loss_collection import Loss
from src.utils.config_utils import (
    get_keychain_value, fraction_to_string, get_kwargs,
    dictionary_to_string)



admissible_model_classes_COREL  = [Autoencoder_MLP.__name__]

@dataclass
class ConfigCOREL:
    __slots__ = ['learning_rate',
                 'batch_size',
                 'n_epochs',
                 'weight_decay',
                 'early_stopping',
                 'rec_loss',
                 'rec_loss_weight',
                 'top_loss',
                 'top_loss_weight',
                 'model_class',
                 'model_kwargs',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid']

    learning_rate: float
    batch_size: int
    n_epochs: int
    weight_decay: float
    early_stopping: int
    rec_loss: Type[Loss]
    rec_loss_weight: float
    top_loss: Type[Loss]
    top_loss_weight: float
    model_class: Type[torch.nn.Module]
    model_kwargs: dict
    dataset: Type[DataSet]
    sampling_kwargs: dict
    eval: ConfigEval
    uid: str


    def __post_init__(self):
        self.check()
        self.uid = self.creat_uuid()

    def creat_uuid(self):
        uuid_suffix = str(uuid.uuid4())[:8]

        uuid_model = '{model}-{hidden_layers}-lr{learning_rate}-bs{batch_size}-nep{n_epochs}-rl{rec_loss_type}{rec_loss_kwargs}-rlw{rec_loss_weight}-tl{top_loss_type}{top_loss_kwargs}-tlw{top_loss_weight}'.format(
            model = self.model_class.__name__,
            hidden_layers = '-'.join(str(x) for x in self.model_kwargs['size_hidden_layers']),
            learning_rate = fraction_to_string(self.learning_rate),
            batch_size = self.batch_size,
            n_epochs = self.n_epochs,
            rec_loss_type = self.rec_loss.__class__.__name__,
            rec_loss_kwargs = get_kwargs(self.rec_loss),
            rec_loss_weight = fraction_to_string(self.rec_loss_weight),
            top_loss_type = self.top_loss.__class__.__name__,
            top_loss_kwargs = get_kwargs(self.top_loss),
            top_loss_weight = fraction_to_string(self.top_loss_weight)
        )

        uuid_data = '{dataset}{object_kwargs}{sampling_kwargs}-'.format(
            dataset = self.dataset.__class__.__name__,
            object_kwargs = get_kwargs(self.dataset),
            sampling_kwargs = dictionary_to_string(self.sampling_kwargs)
        )


        return uuid_data + uuid_model+'-'+uuid_suffix
    
    
    def create_dict(self):
        ret_dict = dict()
        for slot in self.__slots__:
            
            if isinstance(getattr(self, slot), (Loss, DataSet)):
                ret_dict.update({slot: getattr(self, slot).__class__.__name__})
            else:
                ret_dict.update({slot: getattr(self, slot)})

        # need to manually change model class
        ret_dict.update(dict(model_class = self.model_class.__name__))

        return ret_dict

    def create_id_dict(self):

        return dict(
            uid = self.uid,
            learning_rate = self.learning_rate,
            batch_size = self.batch_size,
            n_epochs = self.n_epochs,
            weight_decay = self.weight_decay,
            early_stopping = self.early_stopping,
            rec_loss_weight = self.rec_loss_weight,
            top_loss_weight = self.top_loss_weight,
            top_loss_func=self.top_loss,
        )

    def check(self):
        assert 0 < self.learning_rate
        assert 0 < self.batch_size
        assert 0 < self.n_epochs

        assert self.model_class.__name__ in admissible_model_classes_COREL
        s = inspect.getfullargspec(self.model_class.__init__)
        for a in s.kwonlyargs:
            assert a in self.model_kwargs
        assert self.creat_uuid()
        assert self.create_dict()



@dataclass
class ConfigGrid_COREL:
    __slots__ = ['learning_rate',
                 'batch_size',
                 'n_epochs',
                 'weight_decay',
                 'early_stopping',
                 'rec_loss',
                 'rec_loss_weight',
                 'top_loss',
                 'top_loss_weight',
                 'model_class',
                 'model_kwargs',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid',
                 'experiment_dir',
                 'seed',
                 'verbose']

    learning_rate: List[float]
    batch_size: List[int]
    n_epochs: List[int]
    weight_decay: List[float]
    early_stopping: List[int]
    rec_loss: List[Type[Loss]]
    rec_loss_weight: List[float]
    top_loss: List[Type[Loss]]
    top_loss_weight: List[float]
    model_class: List[Type[torch.nn.Module]]
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
        for slot in (set(self.__slots__)-set(['experiment_dir', 'seed', 'verbose'])):
            grid.update({slot: getattr(self, slot)})
        tmp = list(get_keychain_value(grid))
        values = [x[1] for x in tmp]
        key_chains = [x[0] for x in tmp]

        ret = []

        for v in itertools.product(*values):

            ret_i = {}

            for kc, kc_v in zip(key_chains, v):
                tmp = ret_i
                for k in kc[:-1]:
                    if k not in tmp:
                        tmp[k] = {}

                    tmp = tmp[k]

                tmp[kc[-1]] = kc_v

            ret.append(ConfigCOREL(**ret_i))

        return ret


