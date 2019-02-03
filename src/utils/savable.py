
import json
import pickle
from functools import wraps
from os.path import exists

from src.utils import images
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


def save_tresholds(tresholds):
    fh = open(PATH_TRESHOLDS, 'w+')
    json.dump(tresholds, fh)


def load_tresholds():
    try:
        fh = open(PATH_TRESHOLDS, 'r')
        tresholds = json.load(fh)
    except:
        print("Initializing thresholds")
        tresholds = {}
        names = images.get_possible_logo_names()
        for name in names:
            tresholds[name] = 10
    return tresholds


def reset_thresholds(logos, val):
    thresholds = {logo: val for logo in logos}
    save_tresholds(thresholds)
