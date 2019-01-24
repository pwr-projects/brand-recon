from typing import Mapping

from ..config import *
from ..detector import *
from ..utils import *
from .features import *
from .keypoint import *

# @savable(SAVABLE_LOGO_FEATURES)


def get_logo_features(method, in_grayscale, *logos) -> Mapping[str, Features]:
    logos = logos if logos else get_test_photos_names()
    detector = get_detector(method)

    logo_features = {}
    for logo in tqm(logos, 'Logo features'):
        logo_data = get_logo_photo_by_name(logo, in_grayscale)
        keypoints, descriptors = detector.detectAndCompute(logo_data, None)
        logo_features[logo] = Features(keypoints=list(map(KeyPoint, keypoints)), descriptors=descriptors)

    if len(logos) == 0:
        return logo_features[logos]

    return logo_features
