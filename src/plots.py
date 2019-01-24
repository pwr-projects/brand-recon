import cv2
from .utils import *

def draw_matches(logo, logo_kps, img, img_kps, matches, matches_mask, draw_all=False, show=False):
    flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT if draw_all else cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    img = cv2.drawMatches(logo, logo_kps, img, img_kps, matches,
                           None, matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matches_mask, flags=flags)
    if show:
        img_show(img)              
    return img