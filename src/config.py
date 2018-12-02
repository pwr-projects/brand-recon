try:
    import cv2
except ImportError:
    print('Install OpenCV to proceed.')
    exit(1)

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt


__all__ = [
    'LOGO_EXTS',
    'VERBOSE',
    'LOGO_DIRNAME',
    'LOGO_DIR',
    'TEMP_DIR',
    'DATASET_DIR',
    'LOGO_FEATURES',
    ]

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
