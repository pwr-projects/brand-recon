from enum import Enum, auto
from os.path import join as pj


class KPD(Enum):
    # KEY POINTS DETECTION METHODS
    SIFT = auto()
    ORB = auto()
    SURF = auto()
    BRISK = auto()


class KPM(Enum):
    # KEY POINTS MATCHING METHODS
    FLANN = auto()
    BF = auto()


