import os
from functools import partial

from ..config import *


def exist_assert(path):
    exist = os.path.exists(path)
    assert exist, f'{path} doesn\'t exist'
    return exist


# PATH TESTERS
def isphoto(path: str) -> bool:
    return bool(filename_ext(PHOTO_EXTS).match(path)) # and exist_assert(path)


def isxml(path: str) -> bool:
    return bool(filename_ext(['xml']).match(path)) # and exist_assert(path)
