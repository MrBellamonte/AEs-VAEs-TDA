import inspect
import uuid
from dataclasses import dataclass
from typing import List, Type

import torch

from src.datasets.datasets import DataSet
from src.evaluation.config import ConfigEval
from src.utils.config_utils import fraction_to_string, get_kwargs, dictionary_to_string


@dataclass
class ConfigMLDL:
    __slots__ = ['learning_rate',
                 'batch_size',
                 'n_epochs',
                 'weight_decay',
                 'early_stopping',
                 'loss_ratio',
                 'epsilon',
                 'RegularB',
                 'model_class',
                 'model_kwargs',
                 'mode',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid']
    learning_rate: float
    batch_size: int
    n_epochs: int
    weight_decay: float
    early_stopping: int
    loss_ratio: List[float]
    epsilon: float
    RegularB: float
    model_class: Type[torch.nn.Module]
    model_kwargs: dict
    mode: str
    dataset: Type[DataSet]
    sampling_kwargs: dict
    eval: ConfigEval
    uid: str


    def __post_init__(self):
        self.check()
        self.uid = self.creat_uuid()


    def creat_uuid(self):

        if self.uid == '':

            unique_id = str(uuid.uuid4())[:8]

            uuid_model = 'mode{mode}-{model}-{hidden_layers}-lr{learning_rate}-bs{batch_size}-nep{n_epochs}-loss{loss_ratio}-epsilon{epsilon}'.format(
                mode=self.mode,
                model=self.model_class.__name__,
                hidden_layers='-'.join(str(x) for x in self.model_kwargs['size_hidden_layers']),
                learning_rate=fraction_to_string(self.learning_rate),
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                loss_ratio='-'.join(str(x).replace('.','_') for x in self.loss_ratio),
                epsilon=fraction_to_string(self.epsilon)
            )

            uuid_data = '{dataset}{object_kwargs}{sampling_kwargs}-'.format(
                dataset=self.dataset.__class__.__name__,
                object_kwargs=get_kwargs(self.dataset),
                sampling_kwargs=dictionary_to_string(self.sampling_kwargs)
            )

            return uuid_data+uuid_model+'-'+ unique_id
        else:
            return self.uid

    def check(self):
        assert 0 < self.learning_rate
        assert 0 < self.batch_size
        assert 0 < self.n_epochs

        s = inspect.getfullargspec(self.model_class.__init__)
        for a in s.kwonlyargs:
            assert a in self.model_kwargs
        assert self.creat_uuid()
        assert self.create_dict()
