from collections import namedtuple

import cv2
from tqdm import tqdm

from .config import LOGO_FEATURES
from .utils import get_logo_by_name, get_logo_names, savable

__all__ = [
    'Features',
    'get_logo_features',
]

Features = namedtuple('Features', ['keypoints', 'descriptors'])


@savable(LOGO_FEATURES)
def get_logo_features(*logos: str, verbose=None):
    logos = logos if logos else get_logo_names()
    verbose = verbose if verbose is not None else (len(logos) > 1)

    sift = cv2.xfeatures2d_SIFT.create(nfeatures=None,
                                       nOctaveLayers=None,
                                       contrastThreshold=None,
                                       edgeThreshold=None,
                                       sigma=None)

    logo_features = {}
    with tqdm(logos, 'Logo features', disable=verbose) as lbar:

        logo_data = get_logo_by_name(logo)

        for logo in lbar:
            sift_data = sift.computeAndDetect(logo_data, None)
            logo_features[logo_data] = Features(*sift_data)

    return logo_features
