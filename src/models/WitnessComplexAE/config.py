import inspect
import itertools
import uuid
from dataclasses import dataclass
from typing import Type, List

import torch

from src.datasets.datasets import DataSet
from src.evaluation.config import ConfigEval
from src.models.autoencoder.autoencoders import Autoencoder_MLP_topoae, ConvAE_MNIST
from src.models.loss_collection import Loss
from src.utils.config_utils import (
    get_keychain_value, fraction_to_string, get_kwargs,
    dictionary_to_string, add_default_to_dict)

admissible_model_classes_TopoAE = [Autoencoder_MLP_topoae.__name__, ConvAE_MNIST.__name__]


@dataclass
class ConfigWCAE:
    __slots__ = ['learning_rate',
                 'batch_size',
                 'n_epochs',
                 'weight_decay',
                 'early_stopping',
                 'rec_loss_weight',
                 'top_loss_weight',
                 'toposig_kwargs',
                 'match_edges',
                 'k',
                 'r_max',
                 'model_class',
                 'model_kwargs',
                 'method_args',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid',
                 'seed',
                 'experiment_dir',
                 'device',
                 'num_threads',
                 'verbose',
                 ]
    learning_rate: float
    batch_size: int
    n_epochs: int
    weight_decay: float
    early_stopping: int
    rec_loss_weight: float
    top_loss_weight: float
    match_edges: str
    k: int
    r_max: float
    method_args: dict
    toposig_kwargs: dict
    model_class: Type[torch.nn.Module]
    model_kwargs: dict
    dataset: Type[DataSet]
    sampling_kwargs: dict
    eval: ConfigEval
    uid: str
    seed: int
    experiment_dir : str
    device : str
    num_threads : int
    verbose : bool


    def __post_init__(self):
        if isinstance(self.method_args, dict):
            pass
        else:
            self.method_args = dict()
        self.method_args.update(dict(name = 'topoae_wc',r_max = self.r_max,k = self.k, match_edges = self.match_edges))

        add_default_to_dict(self.sampling_kwargs, 'seed', self.seed)
        add_default_to_dict(self.method_args, 'n_jobs', 1)
        add_default_to_dict(self.method_args, 'verification', False)
        add_default_to_dict(self.method_args, 'wc_offline', None)
        add_default_to_dict(self.method_args, 'normalize', True)
        add_default_to_dict(self.method_args, 'mu_push', 1.05)
        add_default_to_dict(self.method_args, 'dist_x_land', False)
        add_default_to_dict(self.method_args, 'lam_t_bi', None)
        add_default_to_dict(self.method_args, 'lam_t_decay', None)
        add_default_to_dict(self.method_args, 'pre_trained_model', None)
        add_default_to_dict(self.method_args, 'val_size', 0.2)

        self.toposig_kwargs = dict(k=self.k, match_edges=self.match_edges, normalize = self.method_args['normalize'],mu_push = self.method_args['mu_push'])

        self.uid = self.creat_uuid()
        self.check()

    def creat_uuid(self):
        if self.uid == '':

            unique_id = str(uuid.uuid4())[:8]

            if 'size_hidden_layers' in self.model_kwargs:
                if 'lambda_kld' in self.model_kwargs:
                    hidden_layers = '-'.join(str(x) for x in self.model_kwargs['size_hidden_layers'] + [self.model_kwargs['lambda_kld']] )
                else:
                    hidden_layers = '-'.join(
                        str(x) for x in self.model_kwargs['size_hidden_layers'])
            else:
                hidden_layers = 'default'

            uuid_model = '{model}-{hidden_layers}-lr{learning_rate}-bs{batch_size}-nep{n_epochs}-rlw{rec_loss_weight}-tlw{top_loss_weight}-me{match_edges}{mu_push}-k{k}-rmax{r_max}-seed{seed}'.format(
                model=self.model_class.__name__,
                hidden_layers=hidden_layers,
                learning_rate=fraction_to_string(self.learning_rate),
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                rec_loss_weight=fraction_to_string(self.rec_loss_weight),
                top_loss_weight=fraction_to_string(self.top_loss_weight),
                match_edges = self.match_edges,
                mu_push = fraction_to_string(self.toposig_kwargs['mu_push']),
                k = str(self.k),
                r_max = str(int(self.r_max)),
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
        assert 0 < self.learning_rate
        assert 0 < self.batch_size
        assert 0 < self.n_epochs

        #assert self.model_class.__name__ in admissible_model_classes_TopoAE
        s = inspect.getfullargspec(self.model_class.__init__)
        for a in s.kwonlyargs:
            assert a in self.model_kwargs
        assert self.creat_uuid()
        assert self.create_dict()


    def create_id_dict(self):

        return dict(
            uid = self.uid,
            seed=self.seed,
            batch_size = self.batch_size,
        )


@dataclass
class ConfigGrid_WCAE:
    __slots__ = ['learning_rate',
                 'batch_size',
                 'n_epochs',
                 'weight_decay',
                 'early_stopping',
                 'rec_loss_weight',
                 'top_loss_weight',
                 'match_edges',
                 'k',
                 'r_max',
                 'model_class',
                 'model_kwargs',
                 'dataset',
                 'sampling_kwargs',
                 'eval',
                 'uid',
                 'experiment_dir',
                 'seed',
                 'device',
                 'num_threads',
                 'verbose',
                 'toposig_kwargs',
                 'method_args']

    learning_rate: List[float]
    batch_size: List[int]
    n_epochs: List[int]
    weight_decay: List[float]
    early_stopping: List[int]
    rec_loss_weight: List[float]
    top_loss_weight: List[float]
    match_edges: List[str]
    k: List[int]
    r_max: List[float]
    model_class: List[Type[torch.nn.Module]]
    model_kwargs: List[dict]
    dataset: List[Type[DataSet]]
    sampling_kwargs: List[dict]
    eval: List[ConfigEval]
    uid: List[str]
    toposig_kwargs: List[dict]
    method_args: List[dict]
    experiment_dir: str
    seed: int
    device: str
    num_threads: int
    verbose: str


    def configs_from_grid(self):

        grid = dict()

        for slot in (set(self.__slots__)-set(['experiment_dir','seed', 'device', 'num_threads', 'verbose'])):
            grid.update({slot: getattr(self, slot)})
        tmp = list(get_keychain_value(grid))
        values = [x[1] for x in tmp]
        key_chains = [x[0] for x in tmp]

        ret = []

        for v in itertools.product(*values):
            ret_i = {'seed': self.seed,
                     'experiment_dir': self.experiment_dir,
                     'device': self.device,
                     'num_threads': self.num_threads,
                     'verbose': self.verbose}

            for kc, kc_v in zip(key_chains, v):
                tmp = ret_i
                for k in kc[:-1]:
                    if k not in tmp:
                        tmp[k] = {}

                    tmp = tmp[k]

                tmp[kc[-1]] = kc_v

            ret.append(ConfigWCAE(**ret_i))

        return ret

