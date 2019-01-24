#!/bin/python
# %%
from itertools import chain

from src import *
import numpy as np

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

annotations = get_annotations()
# %%
TEST_LOGO = 'windows'
GRAYSCALE = False
photos_with_logos = get_photos_with_logo(TEST_LOGO, annotations)
test_photo = img_load(imgpath(photos_with_logos[12]), GRAYSCALE)
img_show(test_photo)
logo_photo = get_logo_photo_by_name(TEST_LOGO, GRAYSCALE)
# %%

# kpds = [KPD.BRISK, KPD.ORB]
kpds = [KPD.SIFT, KPD.SURF]
detector = Detector(*kpds)
# detector = Detector(KPD.ORB, KPD.BRISK)

logo_feats = detector(logo_photo)
test_photo_feats = detector(test_photo)

# %%
img_show(cv2.drawKeypoints(logo_photo,
                           logo_feats.keypoints,
                           logo_photo))
img_show(cv2.drawKeypoints(test_photo,
                           test_photo_feats.keypoints,
                           logo_photo))
# %%
matcher = Matcher(KPM.FLANN, *kpds)
matches = matcher(logo_feats.descriptors, test_photo_feats.descriptors)
print('Found', len(matches), 'common keypoints')
# %%
img_show(show_matched_logo(matches,
                           logo_feats.keypoints,
                           logo_photo,
                           TEST_LOGO,
                           test_photo_feats.keypoints,
                           test_photo,
                           True,
                           True))


# %%
