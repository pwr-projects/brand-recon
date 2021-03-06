from typing import Sequence

import cv2
import numpy as np

from ..consts import *


class Matcher:
    def __init__(self, kpm: KPM, *kpds: Sequence[KPD], tresh_factor=0.65, print_info=True):
        self._kpds = kpds
        self._kpm = kpm
        self._tresh_factor = tresh_factor
        self._print_info = print_info

    def __call__(self, logo_descs, photo_descs, treshold):
        matcheses = []
        _photo_descs = photo_descs.copy()
        while True:
            matcher_function = self._match(logo_descs, _photo_descs)
            matches = self._filter_matches(matcher_function)
            if len(matches) < treshold:
                #print(f'Found no logo. Only {len(matches)} common keypoints')
                break
            if self._print_info:
                print(f'Found logo with {len(matches)} common keypoints')
                print(f'Remaining {len(_photo_descs)} descriptors')

            matches_ids = list(map(lambda match: match.trainIdx, matches))
            matcheses.append(matches)
            _photo_descs = np.asarray([photo_desc for idx, photo_desc in enumerate(_photo_descs) if idx not in matches_ids])
        return matcheses

    def _filter_matches(self, matches):
        if len(matches) > 0 and hasattr(matches[0], 'distance'):
            matches.sort(key=lambda x: x.distance, reverse=True)
            return list(filter(lambda m: m.distance < 50, matches))
        else:
            matches.sort(key=lambda x: x[0].distance, reverse=True)
            return [match[0] for match in matches
                    if len(match) == 2
                    and match[0].distance < self._tresh_factor * match[1].distance]

    def _match(self, *args):
        return {KPM.BF: self._match_with_bf,
                KPM.FLANN: self._match_with_flann}[self._kpm](*args)

    def _match_with_flann(self, *args):
        if all([kpd in (KPD.SIFT, KPD.SURF) for kpd in self._kpds]):
            index_params = dict(algorithm=0, trees=10)
            search_params = dict(checks=500)

        elif all([kpd in (KPD.BRISK, KPD.ORB) for kpd in self._kpds]):
            index_params = dict(algorithm=6,
                                table_number=15,
                                key_size=20,
                                multi_probe_level=5)
            search_params = dict(checks=500)
        else:
            raise Exception('Pass subset of {SIFT, SURF}, {ORB, BRISK}')

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann.knnMatch(*args, k=2, compactResult=True)

    def _match_with_bf(self, *args):
        if all([kpd in (KPD.SIFT, KPD.SURF) for kpd in self._kpds]):
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            return bf.knnMatch(*args, k=2)
        elif all([kpd in (KPD.BRISK, KPD.ORB) for kpd in self._kpds]):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            return bf.match(*args)
        else:
            raise Exception('Pass subset of {SIFT, SURF}, {ORB, BRISK}')
