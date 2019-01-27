
import pickle
from functools import wraps
from os.path import exists
from os.path import join as pj

from ..config import *


def save(data, path: str):
    with open(path, 'wb') as fhnd:
        pickle.dump(data, fhnd)
    return data


def load(path: str):
    with open(path, 'rb') as fhnd:
        data = pickle.load(fhnd, fix_imports=True)
    return data


def savable(what: str):
    """
    If you want to override the save file simply pass arg:
        override=True 
    in the wrapped function :3
    """
    override_argname = 'override'
    save_load_path = pj(DIR_TEMP, what)

    def real_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (override_argname in kwargs.keys() and kwargs.get(override_argname)) and exists(save_load_path):
                return load(save_load_path)
            return save(func(*args, **kwargs), save_load_path)
        return wrapper
    return real_decorator
