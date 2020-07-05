import inspect
import itertools
import uuid
from fractions import Fraction

from src.model.autoencoders import autoencoder

admissible_model_classes  = [autoencoder.__name__]


def create_model_uuid(config):
    uuid_suffix = str(uuid.uuid4())[:8]

    model = config['model_args']['model_class'].__name__
    hidden_layers = '-'.join(str(x) for x in config['model_args']['kwargs']['size_hidden_layers'])
    learning_rate = str(Fraction(str(config['train_args']['learning_rate']))).replace('/', '_')
    batch_size = config['train_args']['batch_size']
    epochs = config['train_args']['n_epochs']

    rec_loss_type = config['train_args']['rec_loss']['loss_class'].__class__.__name__
    rec_loss_kwargs = ''
    for slot in config['train_args']['rec_loss']['loss_class'].__slots__:
        if type(config['train_args']['rec_loss']['loss_class'].__getattribute__(slot)) == float:
            rec_loss_kwargs += '-'+str(slot)+str(Fraction(str(config['train_args']['rec_loss']['loss_class'].__getattribute__(slot))).replace('/', '_'))
        else:
            rec_loss_kwargs += '-' + str(slot) + str(config['train_args']['rec_loss']['loss_class'].__getattribute__(slot))
    rec_loss_weight = str(Fraction(str(config['train_args']['rec_loss']['weight']))).replace('/', '_')

    top_loss_type = config['train_args']['top_loss']['loss_class'].__class__.__name__
    top_loss_kwargs = ''
    for slot in config['train_args']['top_loss']['loss_class'].__slots__:
        if type(config['train_args']['top_loss']['loss_class'].__getattribute__(slot)) == float:
            top_loss_kwargs += '-'+str(slot)+str(Fraction(str(config['train_args']['top_loss']['loss_class'].__getattribute__(slot))).replace('/', '_'))
        else:
            top_loss_kwargs += '-' + str(slot) + str(config['train_args']['top_loss']['loss_class'].__getattribute__(slot))
    top_loss_weight = str(Fraction(str(config['train_args']['top_loss']['weight']))).replace('/', '_')


    uuid_str = '{}-{}-lr{}-bs{}-nep{}-rl{}{}-rlw{}-tl{}{}-tlw'.format(
        model,
        hidden_layers,
        learning_rate,
        batch_size,
        epochs,
        rec_loss_type,
        rec_loss_kwargs,
        rec_loss_weight,
        top_loss_type,
        top_loss_kwargs,
        top_loss_weight
    )

    return uuid_str+'-'+uuid_suffix


def create_data_uuid(data_args):

    uid = ''
    uid += data_args['dataset'].__class__.__name__

    for slot in data_args['dataset'].__slots__:

        if type(data_args['dataset'].__getattribute__(slot)) == float:
            uid += '-'+str(slot)+str(Fraction(str(data_args['dataset'].__getattribute__(slot)).replace('/', '_')))
        else:
            uid += '-' + str(slot) + str(data_args['dataset'].__getattribute__(slot))

    for key, value in enumerate(data_args['kwargs']):
        if type(value) == float:
            uid += '-'+str(value)+str(Fraction(str(data_args['kwargs'][value]).replace('/', '_')))
        else:
            uid += '-' + str(value) + str(data_args['kwargs'][value])

    return uid




def check_config(config):
    #todo: Think about if it makes sense to create a "config" class....
    #todo: Implement check for rec_loss and top_loss?


    assert 'train_args' in config
    train_args = config['train_args']

    assert 'learning_rate' in train_args
    assert 0 < train_args['learning_rate']

    assert 'batch_size' in train_args
    assert 0 < train_args['batch_size']

    assert 'n_epochs' in train_args
    assert 0 < train_args['n_epochs']

    assert 'rec_loss' in train_args
    assert 'top_loss' in train_args

    # check model-speficic args
    assert 'model_args' in config
    model_args = config['model_args']
    assert 'model_class' in model_args
    assert model_args['model_class'].__name__ in admissible_model_classes
    assert 'kwargs' in model_args
    kwargs = model_args['kwargs']
    s = inspect.getfullargspec(model_args['model_class'].__init__)
    for a in s.kwonlyargs:
        assert a in kwargs

    # check uuid creation
    assert create_model_uuid(config)



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