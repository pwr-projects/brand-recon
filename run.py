# %%
from src import *

# %%


import cv2
import numpy as np


def detect_logo_with_sift(photo_path: str, *logo_names, match_threshold: int = 15, show_match: bool = True):
    detected_logos = []

    sift = cv2.xfeatures2d_SIFT.create(edgeThreshold=200,
                                       sigma=1.6,
                                       nOctaveLayers=7)

    photo = img_load(os.path.join('photos', photo_path + '.jpg'))
    photo_keypoints, photo_descriptors = sift.detectAndCompute(photo, None)

    logo_names = logo_names if logo_names else get_logo_names()
    logos = {k: v for k, v in get_logo_features().items() if k in logo_names}

    for logo_name, (logo_keypoints, logo_descriptors) in tqdm(logos.items(),
                                                              desc='Logo',
                                                              disable=not (len(logo_names) > 1 and VERBOSE)):
        print(logo_name, logo_keypoints, logo_descriptors)
        good_matches = []
        matchesMask = None
        photo_copy = photo.copy()

        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=500))
        matches = flann.knnMatch(logo_descriptors, photo_descriptors, k=2)

        for m, n in matches:
            if m.distance < 0.65 * n.distance:  # TODO make fun with this numeric parameter, it seems to be important
                good_matches.append(m)

        if len(good_matches) > match_threshold:
            print(f'Found logo {logo_name}: {len(good_matches)}/{match_threshold}')
            detected_logos.append(logo_name)

        if show_match:
            pattern = get_logo_by_name(logo_name)
            h, w = pattern.shape[:2]

            src_pts = np.float32([logo_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([photo_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            pts = np.float32([[0, 0],
                              [0, h-1],
                              [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            photo_copy = cv2.polylines(photo_copy, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

            photo_copy = cv2.drawMatches(pattern,
                                         logo_keypoints,
                                         photo_copy,
                                         photo_keypoints,
                                         good_matches,
                                         matchColor=(0, 255, 0),
                                         matchesMask=matchesMask,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            img_show(photo_copy)

    return detected_logos


# %%
# detect_logo_with_sift('windows8', 'windows')


#%%
get_logo_features()


#%%
