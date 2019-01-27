from collections import namedtuple
from typing import Mapping

from ..config import tqm
from ..utils import *

Features = namedtuple('Features', ['keypoints', 'descriptors'])


def get_logo_features(detector, in_grayscale, *logos) -> Mapping[str, Features]:
    logos = logos if logos else get_possible_logo_names()
    return {logo: detector(get_logo_photo_by_name(logo, in_grayscale)) for logo in tqm(logos, 'Creating logo features')}
