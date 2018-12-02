from collections import namedtuple

import cv2

from .config import LOGO_FEATURES, VERBOSE
from .utils import get_logo_by_name, get_logo_names, savable, tqdm
from cv2 import KeyPoint 

__all__ = [
    'Features',
    'get_logo_features',
]

Features = namedtuple('Features', ['keypoints', 'descriptors'])


@savable(LOGO_FEATURES)
def get_logo_features(*logos):
    logos = logos if logos else get_logo_names()
    verbose = len(logos) > 1 and VERBOSE

    sift = cv2.xfeatures2d_SIFT.create(nfeatures=10000,
                                       nOctaveLayers=50,
                                       edgeThreshold=1000,
                                       sigma=1.6)

    logo_features = {}
    for logo in tqdm(logos, 'Logo features', disable=not verbose):
        logo_data = get_logo_by_name(logo)
        keypoints, descriptors = sift.detectAndCompute(logo_data, None)
        logo_features[logo] = Features(keypoints=keypoints, descriptors=descriptors)

    return logo_features
