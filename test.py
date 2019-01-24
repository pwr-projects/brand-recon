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
TEST_LOGO = 'danone'
test_photo = img_load(imgpath(photos_with_logos[1]), False)
img_show(test_photo)
#%%
photos_with_logos = get_photos_with_logo(TEST_LOGO, annotations)
logo_photo = get_logo_photo_by_name(TEST_LOGO, False)

kpds = [KPD.BRISK, KPD.ORB]
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
# %%


def show_matched_logo(good_matches,
                      logo_keypoints,
                      logo,
                      logo_name,
                      photo_keypoints,
                      photo,
                      show_matches,
                      show_detection):

    h, w = logo.shape[:2]

    src_pts = np.float32([logo_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([photo_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)

    if M is not None and M.size != 0:
        dst = cv2.perspectiveTransform(pts, M)
        photo = cv2.polylines(photo, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        # TODO CHECK WHY findHomography sometimes returns empty M
        print('Cannot create this nice matching rect due to... some unknown yet problem :c')
        return

    if show_matches:
        photo = cv2.drawMatches(logo,
                                logo_keypoints,
                                photo,
                                photo_keypoints,
                                good_matches,
                                None,
                                matchColor=(0, 255, 0),
                                singlePointColor=None,
                                matchesMask=matchesMask,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                                #  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
                                )

    if show_detection:
        font = cv2.FONT_HERSHEY_SIMPLEX

        writing_x = int(dst[0][0][0])
        writing_shift = 10
        writing_y = int(min(dst[0][0][1], dst[3][0][1]) - writing_shift)

        bottom_left_corner_of_text = (writing_x, writing_y)
        f_scale, f_type, f_color = 1, 2, (0, 255, 0)

        cv2.putText(photo,
                    logo_name,
                    bottom_left_corner_of_text,
                    font,
                    f_scale,
                    f_color,
                    f_type)

    # photo_copy = cv2.cvtColor(photo_copy, cv2.COLOR_GRAY2RGB)
    return photo


# %%
img_show(show_matched_logo(matches,
                  logo_feats.keypoints,
                  logo_photo,
                  TEST_LOGO,
                  test_photo_feats.keypoints,
                  test_photo,
                  True,
                  True))


#%%
