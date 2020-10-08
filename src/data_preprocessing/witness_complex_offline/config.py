import itertools
import uuid
from dataclasses import dataclass
from typing import Type, List

from src.datasets.datasets import DataSet, SwissRoll
from src.utils.config_utils import get_keychain_value


@dataclass
class ConfigWC:
    __slots__ = [
        'dataset',
        'batch_size',
        'sampling_kwargs',
        'wc_kwargs',
        'eval_size',
        'n_jobs',
        'seed',
        'root_path',
        'global_register',
        'uid',
        'verbose'
    ]
    dataset: Type[DataSet]
    sampling_kwargs: dict
    batch_size: int
    wc_kwargs: dict
    eval_size: float
    n_jobs: int
    seed: str
    global_register: str
    root_path: str
    verbose: bool

    def __post_init__(self):
        self.uid = '{dataset}-bs{batch_size}-seed{seed}-{uid}'.format(
            dataset=self.dataset.__class__.__name__,
            batch_size = self.batch_size,
            seed=self.seed,
            uid=str(uuid.uuid4())[:8]
        )
        self.check()

    def check(self):
        #todo: see if necessary
        pass

    def create_dict(self):
        ret_dict = dict()
        for slot in self.__slots__:

            if isinstance(getattr(self, slot), (DataSet)):
                ret_dict.update({slot: getattr(self, slot).__class__.__name__})
            else:
                ret_dict.update({slot: getattr(self, slot)})

        # need to manually change model class

        return ret_dict


#todo
@dataclass
class ConfigWC_Grid:
    __slots__ = [
        'dataset',
        'sampling_kwargs',
        'batch_size',
        'wc_kwargs',
        'eval_size',
        'n_jobs',
        'seed',
        'root_path',
        'global_register',
        'verbose'
    ]
    dataset: List[Type[DataSet]]
    sampling_kwargs: List[dict]
    batch_size: List[int]
    eval_size: List[float]
    wc_kwargs: List[dict]
    n_jobs: List[int]
    seed: List[str]
    global_register: str
    root_path: str
    verbose: bool

    def configs_from_grid(self):

        grid = dict()

        for slot in (set(self.__slots__)-set(['root_path','global_register', 'verbose'])):
            grid.update({slot: getattr(self, slot)})
        tmp = list(get_keychain_value(grid))
        values = [x[1] for x in tmp]
        key_chains = [x[0] for x in tmp]

        ret = []

        for v in itertools.product(*values):

            ret_i = dict(root_path = self.root_path, global_register = self.global_register, verbose = self.verbose)
            for kc, kc_v in zip(key_chains, v):
                tmp = ret_i
                for k in kc[:-1]:
                    if k not in tmp:
                        tmp[k] = {}

                    tmp = tmp[k]

                tmp[kc[-1]] = kc_v

            ret.append(ConfigWC(**ret_i))

        return ret


placeholder_config_wc = ConfigWC(
    dataset = SwissRoll(),
    sampling_kwargs = dict(n_samples = 2560),
    batch_size=64,
    wc_kwargs=dict(),
    eval_size=0.2,
    n_jobs = 1,
    seed = 1,
    global_register = '',
    root_path = '',
    verbose = False
)


