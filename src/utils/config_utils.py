import itertools
from fractions import Fraction


def get_keychain_value(d, key_chain=None, allowed_values=(list,)):
    key_chain = [] if key_chain is None else list(key_chain).copy()

    if not isinstance(d, dict):
        if allowed_values is not None:
            assert isinstance(d, allowed_values), 'Value needs to be of type {}!'.format(
                allowed_values)
        yield key_chain, d
    else:
        for k, v in d.items():
            yield from get_keychain_value(v, key_chain+[k], allowed_values=allowed_values)


def configs_from_grid(grid):
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

        ret.append(ret_i)

    return ret


def fraction_to_string(fraction: float, delimiter: str = '_'):
    return str(Fraction(str(fraction))).replace('/', delimiter)


def get_kwargs(object):
    ret_str = ''
    for slot in object.__slots__:
        if isinstance(object.__getattribute__(slot),float):
            ret_str += '-'+str(slot)+fraction_to_string(object.__getattribute__(slot))
        else:
            ret_str += '-'+str(slot)+str(object.__getattribute__(slot))

    return ret_str

def dictionary_to_string(dictionary: dict):
    ret_str = ''
    for i, (key, value) in enumerate(dictionary.items()):
        if type(value) == float:
            ret_str += '-'+str(key)+fraction_to_string(value)
        else:
            ret_str += '-' + str(key) + str(value)
    return ret_str


def add_default_to_dict(dict, key, default = False):
    if key in dict:
        pass
    else:
        dict[key] = default



def get_configs(configs, class_config, class_configgrid):
    if isinstance(configs, class_configgrid):
        return configs.configs_from_grid()
    elif isinstance(configs, class_config):
        return [configs]
    else:
        return configs
