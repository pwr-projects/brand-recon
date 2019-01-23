import os
import re
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .misc import exist_assert, tqdm
from ..config import *


def imgpath(path: str) -> str:
    return os.path.join(DIR_DATASET, path)


def img_load(path: str, in_grayscale: bool) -> np.ndarray:
    path = imgpath(path)
    exist_assert(path)
    color_mode = 0 if in_grayscale else 1
    return cv2.imread(path, color_mode)


def img_show(img: np.ndarray, ax=plt, **kwargs) -> plt.Axes:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = ax.imshow(img, **kwargs)
    plt.axis('off')
    plt.show()
    return ax


def get_logo_names(path: bool = False):
    exist_assert(PATH_LOGO)

    logo_names = filter(LOGO_FILENAME_REGEX.match, tqdm(os.listdir(PATH_LOGO),
                                                        desc='Filtering logo files'))
    if path:
        append_path = partial(os.path.join, PATH_LOGO)
        logo_names = map(append_path, tqdm(logo_names,
                                           desc='Appending path'))
    else:
        def remove_ext(name): return ''.join(name.split('.')[:-1])
        logo_names = map(remove_ext, tqdm(logo_names,
                                          desc='Removing extensions'))
    return list(logo_names)


def get_logo_by_name(name: str, in_grayscale=True) -> np.ndarray:

    for ext in PHOTO_EXTS:
        logo_name_with_ext = '.'.join([name, ext])

        if os.path.exists(os.path.join(PATH_LOGO, logo_name_with_ext)):
            return img_load(os.path.join(DIR_LOGO, logo_name_with_ext), in_grayscale)

    raise Exception(f'There\'s no logo of {name}.')


def get_photos_to_test():
    files = os.listdir(PATH_PHOTOS)
    return list(map(lambda filepath: os.path.join(PATH_PHOTOS, filepath), filter(isphoto, files)))
