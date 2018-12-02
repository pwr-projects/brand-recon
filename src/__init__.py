from .config import *
from .detectors import *
from .utils import *

import os

for dirpath in (TEMP_DIR,):
    if os.path.isdir(dirpath):
        os.mkdir(dirpath)
