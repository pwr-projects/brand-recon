from .config import *
from .detectors import *
from .utils import *

import os

# DIRS CREATION
for dirpath in (TEMP_DIR,):
    if not os.path.isdir(dirpath):
        print(f'Creating {dirpath}', end='...')
        os.mkdir(dirpath)
        print('OK')