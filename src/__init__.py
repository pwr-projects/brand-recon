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
for dirpath in (DIR_LOGO,
                DIR_DATASET,
                DIR_TEMP,
                DIR_PHOTOS,
                DIR_ANNOTATIONS,
                DIR_PERSISTED):
    if not os.path.isdir(dirpath):
        print(f'Creating {dirpath}', end='...')
        os.mkdir(dirpath)
        print('OK')
