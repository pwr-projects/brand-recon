import os

from .config import *
from .features import *
from .utils import *
from .core import *

try:
    import cv2
except ImportError:
    print('Install OpenCV to proceed.')
    exit(1)

from colorama import init as color_init
color_init()

# DIRS CREATION
for dirpath in (PATH_LOGO,
                PATH_ANNOTATIONS,
                PATH_PHOTOS,
                PATH_PERSISTED,
                PATH_TRESHOLDS,
                DIR_TEMP):
    if not os.path.isdir(dirpath):
        print(f'Creating {dirpath}', end='...')
        os.mkdir(dirpath)
        print('OK')
