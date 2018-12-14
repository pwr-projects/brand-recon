try:
    import cv2
except ImportError:
    print('Install OpenCV to proceed.')
    exit(1)

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, auto

# KEY POINTS DETECTION METHODS
class KPD(Enum):
    SIFT = auto()
    ORB = auto()
    SURF = auto()
    BRISK = auto()

METHODS = list(KPD)

# KEY POINTS MATCHING METHODS
class KPM(Enum):
    FLANN = auto()
    BF = auto()

MATCHING_METHODS = list(KPM)

# MATPLOTLIB SETTINGS
plt.style.use('seaborn-paper')
mpl.rcParams['figure.figsize'] = [16.0, 9.0]
mpl.rcParams['figure.dpi'] = 150

# UTILS
LOGO_EXTS = ['jpg', 'png']
VERBOSE = True
# FILES
LOGO_DIRNAME = 'logos'

DATASET_DIR = 'data'
TEMP_DIR = '.tmp'
LOGO_DIR = os.path.join(DATASET_DIR, LOGO_DIRNAME)

# SAVABLE
LOGO_FEATURES = 'logo_features'
