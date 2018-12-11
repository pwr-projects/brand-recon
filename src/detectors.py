from collections import namedtuple

import cv2

from .config import VERBOSE, METHODS
from .utils import get_logo_by_name, get_logo_names, tqdm

__all__ = [
    'Features',
    'get_logo_features',
]

Features = namedtuple('Features', ['keypoints', 'descriptors'])


@savable(LOGO_FEATURES)
def get_logo_features(method, *logos):
    logos = logos if logos else get_logo_names()
    verbose = len(logos) > 1 and VERBOSE

    detector = get_detector(method)

    logo_features = {}
    for logo in tqdm(logos, 'Logo features', disable=not verbose):
        logo_data = get_logo_by_name(logo)
        keypoints, descriptors = detector.detectAndCompute(logo_data, None)
        logo_features[logo] = Features(keypoints=keypoints, descriptors=descriptors)

    return logo_features


def get_detector(method):
    if method == 'SIFT':
        detector = cv2.xfeatures2d_SIFT.create(nfeatures=10000,
                                               nOctaveLayers=50,
                                               edgeThreshold=1000,
                                               sigma=1.6)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=1500,
                                  edgeThreshold=25,
                                  scaleFactor=1.2,
                                  nlevels=8)
    elif method == 'SURF':
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=100,
                                               nOctaves=4,
                                               nOctaveLayers=3)
    elif method == 'BRISK':
        detector = cv2.BRISK_create()
    else:
        raise ValueError("Method must be from:" + str(METHODS))
    return detector
