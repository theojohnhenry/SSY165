import inspect
from .datastructures import Transition


def filter_trans_by_source(trans, states_to_keep):
    """Returns a new set containing all transitions where the source is in states_to_keep"""
    return {t for t in trans if t.source in states_to_keep}


def filter_trans_by_events(trans, events_to_keep):
    """Returns a new set containing all transitions where the event is in events_to_keep"""
    return {t for t in trans if t.event in events_to_keep}


def filter_trans_by_target(trans, states_to_keep):
    """Returns a new set containing all transitions where the target is in states_to_keep"""
    return {t for t in trans if t.target in states_to_keep}


def extract_elems_from_trans(trans, field):
    """
    Returns a new set with just the elements in a field of all transitions.
    E.g. field='source' for all source states
    or field='event' or field='target'
    """
    return {getattr(t, field) for t in trans}


def flip_trans(trans):
    """Flips the direction of the transitions in the set"""
    return {Transition(t.target, t.event, t.source) for t in trans}


def merge_label(label1, label2):
    """Creates a new label based on two labels"""
    return '{}.{}'.format(label1, label2)


def cross_product(setA, setB):
    """Computes the crossproduct of two sets"""
    return {merge_label(a, b) for b in setB for a in setA}


def validate_grid_config(env_config, param_grid):
    """Validates hyperparameter grid for RL"""
    title_items = [env_config['name']]
    X, Y = None, None
    for attr_str in ['epsilon', 'gamma', 'A', 'B', 'pripps_reward']:
        attr = getattr(param_grid, attr_str)
        attr_str = '$\{}$'.format(attr_str) if attr_str in ['epsilon', 'gamma'] else attr_str
        if type(attr) is not list or len(attr) < 1:
            raise ValueError('Each parameter of the HyperParameterGrid must be given as a list of length >= 1')
        elif len(attr) == 1:
            title_items += [attr_str, attr[0]]
        elif len(attr) > 1 and X is None:
            X = sorted(attr)
            x_name = attr_str
        elif len(attr) > 1 and Y is None:
            Y = sorted(attr)
            y_name = attr_str
        else:
            raise ValueError('Only 2 parameter of the HyperParameterGrid can be given as a list of length > 1')
    assert X is not None and Y is not None, 'Please choose 2 axes for the grid by passing lists of lenght > 1 for 2 parameters.'
    if param_grid.pripps_reward[0]:
        title = '{}: Learning Stats for {} = {}, {} = {}, and {} = {}'.format(*title_items)
    else:
        title = '{}: Learning Stats for {} = {} and {} = {}'.format(*title_items)
    return X, Y, x_name, y_name, title


def write_defs_to_file(initialize_Q, argmax_Q, choose_epsilon_greedily, get_alpha, learn_q, eval_learning_params, file_name):
    """Workaround for multiprocessing in Jupyter on Windows"""
    defs = [inspect.getsource(f) for f in [initialize_Q, argmax_Q, choose_epsilon_greedily, get_alpha]]
    defs.append(inspect.getsource(learn_q).replace(learn_q.__name__, "learning_f"))
    defs.append(inspect.getsource(eval_learning_params).replace(eval_learning_params.__name__, "task"))
    with open(file_name, 'w') as file:
        file.write('import random')
        file.write('\n')
        file.write('import numpy as np')
        for f in defs:    
            file.write('\n\n')
            file.write(f)