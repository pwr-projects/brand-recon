import os
import re
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .misc import exist_assert, tqdm
from ..config import *

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


def img_load(path: str) -> np.ndarray:
    path = imgpath(path)
    exist_assert(path)
    return cv2.imread(path, 0)


def img_show(img: np.ndarray, ax=plt, **kwargs):
    ax = ax.imshow(img, **kwargs)
    plt.axis('off')
    plt.show()
    return ax


def get_logo_names(path: bool = False):
    exist_assert(LOGO_DIR)
    logo_names = filter(LOGO_FILENAME_REGEX.match, tqdm(os.listdir(LOGO_DIR),
                                                        desc='Filtering logo files'))
    if path:
        append_path = partial(os.path.join, LOGO_DIR)
        logo_names = map(append_path, tqdm(logo_names,
                                           desc='Appending path'))
    else:
        def remove_ext(name): return ''.join(name.split('.')[:-1])
        logo_names = map(remove_ext, tqdm(logo_names,
                                          desc='Removing extensions'))
    return list(logo_names)


def get_logo_by_name(name: str) -> np.ndarray:

    for ext in LOGO_EXTS:
        logo_name_with_ext = '.'.join([name, ext])

        if os.path.exists(os.path.join(LOGO_DIR, logo_name_with_ext)):
            return img_load(os.path.join(LOGO_DIRNAME, logo_name_with_ext))

    raise Exception(f'There\'s no logo of {name}.')
