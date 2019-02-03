from enum import Enum, auto


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


class TM(Enum):
    # THRESHOLDING METHODS
    CONSTANT = auto()
    PROGRESSIVE = auto()
    OPTIMIZED = auto()


class PREPROC(Enum):
    # PREPROCESSING METHODS
    NONE = auto()
    BINARIZATION = auto()
    HIST_EQ = auto()

