from collections import namedtuple

import cv2

from .config import *
from .utils import get_logo_by_name, get_logo_names, tqdm, savable

__all__ = [
    'Features',
    'get_logo_features',
]

Features = namedtuple('Features', ['keypoints', 'descriptors'])


# @savable(LOGO_FEATURES)
def get_logo_features(method, in_grayscale, *logos):
    logos = logos if logos else get_logo_names()
    verbose = len(logos) > 1 and VERBOSE

    detector = get_detector(method)

    logo_features = {}
    for logo in tqdm(logos, 'Logo features'):
        logo_data = get_logo_by_name(logo, in_grayscale)
        keypoints, descriptors = detector.detectAndCompute(logo_data, None)
        logo_features[logo] = Features(keypoints=keypoints, descriptors=descriptors)

    return logo_features


def get_detector(method):
    method_funcs = {
        KPD.SIFT: lambda: cv2.xfeatures2d_SIFT.create(nOctaveLayers=40,
                                                      edgeThreshold=500,
                                                      contrastThreshold=0.03,
                                                      sigma=1.6),

        KPD.ORB: lambda: cv2.ORB_create(nfeatures=1500,
                                        edgeThreshold=10,
                                        scaleFactor=1.2,
                                        nlevels=8),

        KPD.SURF: lambda: cv2.xfeatures2d_SURF.create(hessianThreshold=300,
                                                      nOctaves=4,
                                                      nOctaveLayers=4,
                                                      extended=True),

        KPD.BRISK: lambda: cv2.BRISK_create(thresh=10, octaves=2)
    }

    return method_funcs[method]()
