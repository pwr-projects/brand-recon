import os

from ..config import *

__all__ = [
    'exist_assert',
]


def exist_assert(path):
    assert os.path.exists(path), f'{path} doesn\'t exist'
