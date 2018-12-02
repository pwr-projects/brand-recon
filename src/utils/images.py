import os
import re

import matplotlib.pyplot as plt

from ..config import *
from .misc import exist_assert

__all__ = [
    'img_load',
    'img_show',
    'get_logo_names',
    'get_logo_by_name',
]

LOGO_EXTS_ALT = '|'.join(LOGO_EXTS)
LOGO_FILENAME_REGEX = re.compile(r'^\w+\.(' + LOGO_EXTS_ALT + r')$')


def imgpath(path: str) -> str:
    return os.path.join(DATASET_DIR, path)


def img_load(path: str):
    path = imgpath(path)
    exist_assert(path)
    return plt.imread(path)


def img_show(img, ax=plt, **kwargs):
    ax = ax.imshow(img, **kwargs)
    plt.axis('off')
    plt.show()
    return ax


def get_logo_names(path: bool = False):
    exist_assert(LOGO_DIR)
    append_path = partial(os.path.join, LOGO_DIR)
    logo_names = filter(LOGO_FILENAME_REGEX.match, os.listdir(LOGO_DIR))
    if path:
        logo_names = map(append_path, logo_names)
    else:
        logo_names = map(lambda name: ''.join(name.split('.')[:-1]), logo_names)
    return list(logo_names)


def get_logo_by_name(name):
    for ext in LOGO_EXTS:
        path_to_check = os.path.join(LOGO_DIRNAME, '.'.join([name, ext]))

        if os.path.exists(os.path.join(DATASET_DIR, path_to_check)):
            return img_load(path_to_check)

    raise Exception(f'There\'s no logo of {name}.')
