import re
from functools import partial
from os.path import join as pj

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

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
DIR_PERSISTED = 'persisted'

# PATHS
PATH_LOGO = pj(DIR_DATASET, DIR_LOGO)
PATH_PHOTOS = pj(DIR_DATASET, DIR_PHOTOS)
PATH_ANNOTATIONS = pj(DIR_DATASET, DIR_ANNOTATIONS)
PATH_TRESHOLDS = pj(DIR_DATASET, DIR_PERSISTED, 'tresholds.json')

# SAVABLE
SAVABLE_LOGO_FEATURES = 'logo_features'

# PHOTO FILENAME REGEX
filename_ext = lambda exts: re.compile(r'^\w+\.(' + '|'.join(exts) + r')$')

LOGO_FILENAME_REGEX = filename_ext(PHOTO_EXTS)

# custom tqdm
tqm = partial(tqdm, disable=not VERBOSE, dynamic_ncols=True)
