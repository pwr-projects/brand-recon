from collections import defaultdict
from typing import Sequence

import cv2
import numpy as np

from .annotations import *
from .misc import *


def imgpath(filename: str) -> str:
    return os.path.join(PATH_PHOTOS, filename)


def logopath(filename: str) -> str:
    return os.path.join(PATH_LOGO, filename)


def img_load(path: str, in_grayscale: bool) -> np.ndarray:
    exist_assert(path)
    color_mode = 0 if in_grayscale else 1
    return cv2.imread(path, color_mode)


def img_show(img: np.ndarray, ax=plt, **kwargs) -> plt.Axes:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # imread(img, 0)
    ax = ax.imshow(img, **kwargs)
    plt.axis('off')
    plt.show()
    return ax


def get_possible_logo_names(path: bool = False) -> Sequence[str]:
    exist_assert(PATH_LOGO)

    logo_names = filter(LOGO_FILENAME_REGEX.match, tqm(os.listdir(PATH_LOGO),
                                                       desc='Filtering logo files'))
    if path:
        append_path = partial(os.path.join, PATH_LOGO)
        logo_names = map(append_path, tqm(logo_names,
                                          desc='Appending path'))
    else:
        remove_ext = lambda name: ''.join(name.split('.')[:-1])
        logo_names = map(remove_ext, tqm(logo_names,
                                         desc='Removing extensions'))
    return list(logo_names)


def get_logo_photo_by_name(name: str, in_grayscale=True) -> np.ndarray:

    for ext in PHOTO_EXTS:
        logo_name_with_ext = '.'.join([name, ext])

        if os.path.exists(logopath(logo_name_with_ext)):
            return img_load(logopath(logo_name_with_ext), in_grayscale)

    raise Exception(f'There\'s no logo of {name}.')


def get_test_photos_names() -> Sequence[str]:
    files = os.listdir(PATH_PHOTOS)
    return list(map(lambda filepath: os.path.join(PATH_PHOTOS, filepath),
                    filter(isphoto, files)))


def get_photos_with_logo(logo: str, annotations: Mapping[str, str] = None) -> Sequence[str]:
    # assert logo in get_possible_logo_names(), 'There is no such logo available'
    if not annotations:
        annotations = get_annotations()

    reversed_annotations = defaultdict(list)
    for k, v in annotations.items():
        reversed_annotations[v].append(k)
    return reversed_annotations[logo]
