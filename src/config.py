try:
    import cv2
except ImportError:
    print('Install OpenCV to proceed.')
    exit(1)

from tqdm import tqdm
from functools import partial


import re
from enum import Enum, auto
from os.path import join as pj

import matplotlib as mpl
import matplotlib.pyplot as plt

from .consts import KPD, KPM

# UTILS
PHOTO_EXTS = ['jpg', 'png']
VERBOSE = True

METHODS = list(KPD)
MATCHING_METHODS = list(KPM)

# MATPLOTLIB SETTINGS
plt.style.use('seaborn-paper')
mpl.rcParams['figure.figsize'] = [16.0, 9.0]
mpl.rcParams['figure.dpi'] = 150


# FILES
DIR_LOGO = 'logos'
DIR_DATASET = 'data'
DIR_TEMP = '.tmp'
DIR_PHOTOS = 'photos'
DIR_ANNOTATIONS = 'annotations'

# PATHS
PATH_LOGO = pj(DIR_DATASET, DIR_LOGO)
PATH_PHOTOS = pj(DIR_DATASET, DIR_PHOTOS)
PATH_ANNOTATIONS = pj(DIR_DATASET, DIR_ANNOTATIONS)

# SAVABLE
SAVABLE_LOGO_FEATURES = 'logo_features'

# PHOTO FILENAME REGEX
filename_ext = lambda exts: re.compile(r'^\w+\.(' + '|'.join(exts) + r')$')

LOGO_FILENAME_REGEX = filename_ext(PHOTO_EXTS)

# custom tqdm
tqdm = partial(tqdm, disable=not VERBOSE, dynamic_ncols=True)
