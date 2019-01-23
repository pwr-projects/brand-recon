import os
from functools import partial

from tqdm import tqdm

from ..config import *


def exist_assert(path):
    assert os.path.exists(path), f'{path} doesn\'t exist'


# PATH TESTERS
def isphoto(path: str) -> bool:
    return bool(filename_ext(PHOTO_EXTS).match(path))


def isxml(path: str) -> bool:
    return bool(filename_ext(['xml']).match(path))
