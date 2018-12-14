import os
from functools import partial

from tqdm import tqdm

from ..config import *


def exist_assert(path):
    assert os.path.exists(path), f'{path} doesn\'t exist'


tqdm = partial(tqdm, disable=not VERBOSE)
