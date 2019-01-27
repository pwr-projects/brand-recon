from itertools import chain

import numpy as np

from ..config import *
from ..features import *
from ..utils import *
from .detector import Detector
from .matcher import Matcher


class LogoDetector:
    def __init__(self, detector: Detector, matcher: Matcher, in_grayscale: bool = False):
        self._detector = detector
        self._matcher = matcher
        self._grayscale = in_grayscale
        self._logo_names = get_possible_logo_names()
        self._logo_feats = get_logo_features(self._detector,
                                             self._grayscale,
                                             *self._logo_names)

    def __call__(self, photo, treshold: int):
        detected_logos = {}

        if type(photo) is str:
            photo = img_load(imgpath(photo), self._grayscale)

        photo_features = self._detector(photo)
        for logo_name, (logo_feats_kps, logo_feats_descs) in tqdm(self._logo_feats.items(), 'Founding logo'):
            print(np.asarray(logo_feats_kps).shape, np.asarray(photo_features.keypoints).shape)
            matches = self._matcher(logo_feats_kps, photo_features.keypoints)
            if len(matches) > treshold:
                print('Found', logo_name)
                detected_logos[logo_name] = len(good_matches)
        return detected_logos
