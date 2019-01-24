import os

from .config import *
from .features import *
from .utils import *
from .plots import *
from .detector import *
from .matcher import *

try:
    import cv2
except ImportError:
    print('Install OpenCV to proceed.')
    exit(1)

# DIRS CREATION
for dirpath in (DIR_TEMP,):
    if not os.path.isdir(dirpath):
        print(f'Creating {dirpath}', end='...')
        os.mkdir(dirpath)
        print('OK')
