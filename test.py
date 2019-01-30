#!/bin/python
# %%

from src import *
from src.core import Detector

print(Fore.BLUE, " ### Started test")

print(Fore.BLUE, " #$ Start: Getting annotations")
annotations = get_annotations()
print(Fore.BLUE, " #! End: Getting annotations\n")

# kpds = [KPD.ORB, KPD.BRISK]
kpds = [KPD.SIFT, KPD.SURF]

detector = Detector(*kpds)
matcher = Matcher(KPM.FLANN, *kpds)
# %%
TEST_LOGO = 'lot'
GRAYSCALE = False

found_logos = {}
print(Fore.BLUE, " #$ Start: Getting photos with logo")
photos_with_logos = get_photos_with_logo(TEST_LOGO, annotations)[:]
print(Fore.BLUE, " #! End: Getting photos with logo\n")

print(Fore.BLUE, " #$ Start: Getting logo photo by name")
logo_photo = get_logo_photo_by_name(TEST_LOGO, GRAYSCALE)
print(Fore.BLUE, " #! End: Getting logo photo by name\n")

print(Fore.BLUE, " #$ Start: Getting features of logo")
logo_feats = detector(logo_photo)
print(Fore.BLUE, " #! End: Getting features of logo\n")


def test(test_photo_name):
    test_photo_path = imgpath(test_photo_name)
    test_photo = img_load(test_photo_path, GRAYSCALE)
    # img_show(test_photo)
    test_photo_feats = detector(test_photo)

    treshold = len(logo_feats.keypoints) * 0.015
    print(" -- # Treshold:", treshold)
    matcheses = matcher(logo_feats.descriptors, test_photo_feats.descriptors, treshold)
    print('Found', len(matcheses), 'logos in:', test_photo_name)

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


# %%
print(Fore.BLUE, " #$ Start: Finding logos")
# with mp.Pool(mp.cpu_count()) as pool:
#     found_logos = {val[0]: (*val[1:],) for val in pool.map(test, photos_with_logos)}
found_logos = []
for photo in photos_with_logos:
    try:
        fl = test(photo)
        found_logos.append(fl)
    except Exception:
        print("Exception occured")
found_logos = {val[0]: (*val[1:],) for val in found_logos}
print(Fore.BLUE, " #! End: Finding logos\n")

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
