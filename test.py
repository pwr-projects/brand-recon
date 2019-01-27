#!/bin/python
# %%
import multiprocessing as mp
from itertools import chain

import numpy as np

from src import *

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

annotations = get_annotations()

# kpds = [KPD.ORB, KPD.BRISK]
kpds = [KPD.SIFT, KPD.SURF]

detector = Detector(*kpds)
matcher = Matcher(KPM.FLANN, *kpds)
# %%
TEST_LOGO = 'biedronka'
GRAYSCALE = False

found_logos = {}
photos_with_logos = get_photos_with_logo(TEST_LOGO, annotations)[:3]
logo_photo = get_logo_photo_by_name(TEST_LOGO, GRAYSCALE)
logo_feats = detector(logo_photo)

    
def test(test_photo_name):
    test_photo_path = imgpath(test_photo_name)
    test_photo = img_load(test_photo_path, GRAYSCALE)
    # img_show(test_photo)
    test_photo_feats = detector(test_photo)

    matcheses = matcher(logo_feats.descriptors, test_photo_feats.descriptors, len(logo_feats.keypoints) * 0.03)
    print('Found', len(matcheses), 'logos')

    test_photo_copy = test_photo.copy()
    for matches in matcheses:
        test_photo_copy = show_matched_logo(matches,
                                            logo_feats.keypoints,
                                            logo_photo,
                                            TEST_LOGO,
                                            test_photo_feats.keypoints,
                                            test_photo_copy,
                                            True,
                                            True)
    # img_show(test_photo_copy)
    return test_photo_path, (test_photo_copy, len(matcheses))
#%%
with mp.Pool(mp.cpu_count()) as pool:
    found_logos = {val[0]: (*val[1:],) for val in pool.map(test, photos_with_logos)}


# %%
for img in found_logos.values():
    img_show(img[0][0])

print(f'Found {len(found_logos.keys())}/{len(photos_with_logos)} logos')

# TODO rozkminić rozmycie
# img_show(cv2.drawKeypoints(logo_photo.copy(),
#                            logo_feats.keypoints,
#                            None))
# img_show(cv2.drawKeypoints(test_photo.copy(),
#                            test_photo_feats.keypoints,
#                            None))

# %%
# logo_detector = LogoDetector(detector, matcher)
# logo_detector(test_photo, 30)


# %%
