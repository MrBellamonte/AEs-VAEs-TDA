import numpy as np

def avg_array_in_dict(a_dict):
    new_dict = dict()
    for key, value in a_dict.items():
        if isinstance(value, np.ndarray):
            new_dict.update(
                {
                    key : np.mean(value)
                }
            )
        elif isinstance(value, list):
            new_dict.update(
                {
                    key : sum(value)/len(value)
                }
            )
        else:
            new_dict.update(
                {
                    key: value
                }
            )
    return new_dict


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))