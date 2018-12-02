# %%

from src import *

# %%


import cv2
from tqdm import tqdm
import numpy as np

def detect_with_sift(photo_path: str, logo_name: str = None, match_threshold=15):

    sift = cv2.xfeatures2d_SIFT.create(edgeThreshold=200,
                                       sigma=1.6,
                                       nOctaveLayers=7)

    img2 = img_load(os.path.join('photos', photo_path + '.jpg'))
    kp2, des2 = sift.detectAndCompute(img2, None)

    logos = [logo_name, ] if logo_name else get_logo_names()
    detected_logos = []

    for logo in tqdm(logos, desc='Logo', disable=len(logos) == 1):
        pattern = get_logo_by_name(logo)

        kp1, des1 = sift.detectAndCompute(pattern, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=50), dict(checks=500))
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)

        img = img2.copy()
        matchesMask = None
        if len(good) > match_threshold:
            print(f'Found logo {logo}: {len(good)}/{match_threshold}')

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = pattern.shape[:2]

            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

            detected_logos.append(logo)
        else:
            print(f'{logo} detected with {np.round((100 * len(good))/match_threshold)}% prob')

        img = cv2.drawMatches(pattern, kp1, img, kp2, good,
                              None, (0, 255, 0), None, matchesMask,
                              cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img_show(img)
    return detected_logos


# %%
detect_with_sift('windows8', 'windows')


#%%
