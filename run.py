#!/usr/bin/python
# %%

import json

import numpy as np
from colorama import Fore, Style

from src import *


#init()

# %%


def detect_logo(photo_name: str,
                *logo_names,
                tresholds,
                detection_method: str = KPD.SIFT,
                matching_method: str = KPM.FLANN,
                match_threshold: int = 50,
                show_match: bool = True,
                show_detection: bool = False,
                in_grayscale=True) -> dict:
    """ Function detecting logos in given photo.

    Args:
        photo_name:       Path to the input photo where the logos are gona be looked for. Example: 'adidas_shoe.png'
        logo_names:       List of logos to look for.
        detection_method: Method of feature extraction from photo.
        matching_method:  Method of key points matching.
        match_threshold:  Threshold of common key points for determining of specific logo presence.
        show_match:       If there will be displayed output image showing matching.
        show_detection:   If there will be displayed bounding box and writing for logo.

    Returns:
        detected_logos: List of detected logo names.

    """

    detected_logos = {}
    detected_logos_imgs = {}

    # photo_path = os.path.join('photos', photo_name)
    photo_path = photo_name
    photo = img_load(photo_path, in_grayscale=in_grayscale)
    photo_keypoints, photo_descriptors = detect_features(detection_method, photo)

    logos = get_logos_with_features(detection_method, *logo_names)
    print("\nFound", len(logos), "logos to compare to.")

    for logo_name, (logo_keypoints, logo_descriptors) in logos:
        print("\nComparing input photo having", len(photo_keypoints), "keypoints, with logo '", logo_name, "' having",
              len(logo_keypoints), "keypoints.")

        good_matches = create_good_matches(matching_method,
                                           detection_method,
                                           logo_descriptors,
                                           photo_descriptors)

        if not logo_name in tresholds.keys():  # TODO: move to other place and then remove match_treshold from arguments
           tresholds[logo_name] = match_threshold

        match_threshold = tresholds[logo_name]

        print("\nNumber of good matches between input and logo '", Fore.GREEN if len(good_matches) >
              match_threshold else Fore.RED, logo_name, Style.RESET_ALL, "' is equal to", len(good_matches))

        if len(good_matches) > match_threshold:
            print(f'Found logo {logo_name}: {len(good_matches)}/{match_threshold}')

            img = show_matched_logo(good_matches,
                                    logo_keypoints,
                                    logo_name,
                                    photo_keypoints,
                                    photo_path,
                                    show_match,
                                    show_detection,
                                    in_grayscale)

            detected_logos[logo_name] = len(good_matches)  # , img

    return detected_logos


def show_matched_logo(good_matches,
                      logo_keypoints,
                      logo_name,
                      photo_keypoints,
                      photo_path,
                      show_matches,
                      show_detection,
                      in_grayscale=False):

    photo_copy = img_load(photo_path, in_grayscale=in_grayscale)
    pattern = get_logo_by_name(logo_name, in_grayscale=in_grayscale)

    h, w = pattern.shape[:2]

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
        photo_copy = cv2.polylines(photo_copy, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        # TODO CHECK WHY findHomography sometimes returns empty M
        print('Cannot create this nice matching rect due to... some unknown yet problem :c')
        return

    if show_matches:
        photo_copy = cv2.drawMatches(pattern,
                                     logo_keypoints,
                                     photo_copy,
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

        cv2.putText(photo_copy,
                    logo_name,
                    bottom_left_corner_of_text,
                    font,
                    f_scale,
                    f_color,
                    f_type)

    # photo_copy = cv2.cvtColor(photo_copy, cv2.COLOR_GRAY2RGB)
    if show_matches or show_detection:
        img_show(photo_copy)
    return photo_copy


def create_good_matches(matching_method, detection_method, logo_descriptors, photo_descriptors):
    matches = match_descriptors(matching_method, detection_method, logo_descriptors, photo_descriptors)

    if len(matches) > 0 and hasattr(matches[0], 'distance'):
        matches.sort(key=lambda x: x.distance, reverse=True)
        return [m for m in matches if m.distance < 50]
    else:
        # matches.sort(key=lambda x: x[0].distance, reverse=True)
        return [match[0] for match in matches
                if len(match) == 2
                and match[0].distance < 0.7 * match[1].distance]


def match_descriptors(method, detection_method, logo_descriptors, photo_descriptors):
    if method == KPM.FLANN:
        if detection_method in (KPD.SIFT, KPD.SURF):
            index_params = dict(algorithm=0, trees=50)
            search_params = dict(checks=500)

        elif detection_method in (KPD.ORB, KPD.BRISK):
            index_params = dict(algorithm=6,
                                table_number=15,  # 12
                                key_size=20,  # 20
                                multi_probe_level=2)  # 2
            search_params = dict(checks=500)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(logo_descriptors, photo_descriptors, k=2, compactResult=True)
        # matches = flann.radiusMatch(logo_descriptors, photo_descriptors, 100)

    elif method == KPM.BF:
        if detection_method in (KPD.SIFT, KPD.SURF):
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(logo_descriptors, photo_descriptors, crossCheck=True, k=2)
        elif detection_method in (KPD.ORB, KPD.BRISK):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(logo_descriptors, photo_descriptors)
    else:
        raise ValueError("Method must be from:" + str(MATCHING_METHODS))
    return matches


def get_logos_with_features(method, *logo_names):
    logo_names = logo_names if logo_names else get_logo_names()
    logos = get_logo_features(method, False, *logo_names)
    # logos = tqdm(logos.items(), desc='Logo', disable=not (len(logo_names) > 1 and VERBOSE))
    return logos.items()


def detect_features(method, photo):
    method_funcs = {
        KPD.SIFT: detect_features_using_sift,
        KPD.ORB: detect_features_using_orb,
        KPD.SURF: detect_features_using_surf,
        KPD.BRISK: detect_features_using_brisk,
    }
    return reversed(method_funcs[method](photo))


def detect_features_using_brisk(photo):
    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(photo, None)
    return descriptors, keypoints


def detect_features_using_surf(photo):
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=100,
                                       nOctaves=4,
                                       nOctaveLayers=4,
                                       extended=True)
    keypoints, descriptors = surf.detectAndCompute(photo, None)
    return descriptors, keypoints


def detect_features_using_orb(photo):
    orb = cv2.ORB_create(  # nfeatures=1500,
        edgeThreshold=20,
        scaleFactor=1.6,
        nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(photo, None)
    return descriptors, keypoints


def detect_features_using_sift(photo):
    sift = cv2.xfeatures2d_SIFT.create(  # nfeatures=1500,
        nOctaveLayers=8,
        edgeThreshold=10,
        contrastThreshold=0.03,
        sigma=1.6)
    keypoints, descriptors = sift.detectAndCompute(photo, None)
    return descriptors, keypoints


def run_in_loop(**kwargs):
    print("### Program started ###")
    action = input("# Type:\n - 'c' to close the program\n - 'name_of_file' to detect logos in file: ")
    while action != 'c':
        print("# Started logo detection")
        detect_logo(action, *get_logo_names(), **kwargs)
        print("# Finished logo detection")
        action = input("# Type:\n - 'c' to close the program\n - 'name_of_file' to detect logos in file: ")
    print("### Closing program ###")


def save_tresholds(tresholds):
    fh = open("data\\persisted\\tresholds.json", 'w+') # TODO: find a better name for the dir to save the tresholds
    json.dump(tresholds, fh)


def load_tresholds():
    try:
        fh = open("data\\persisted\\tresholds.json", 'r')
        tresholds = json.load(fh)
    except:
        tresholds = {}
        names = get_logo_names()
        for name in names:
            tresholds[name] = 25
    return tresholds


def test_all_from_dir(directory_in_str, tresholds, should_print_info=False):
    logo_to_detect = os.path.basename(directory_in_str)
    fn_images_list = []

    local_cms = init_confusion_matrices()

    for file in os.listdir(directory_in_str):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"): # or filename.endswith(".png"):
            file_path = os.path.join('all_pics', logo_to_detect, filename)
            dl = detect_logo(
                file_path,
                *get_logo_names(),
                tresholds=tresholds,
                detection_method=KPD.ORB,
                matching_method=KPM.FLANN,
                match_threshold=35,
                show_match=False,
                show_detection=False,
                in_grayscale=False)

            ''' if not dl == {} and logo_to_detect == max(dl, key=dl.get):  # in dl.keys():
                # TP
                tp += 1
            else:
                if not dl == {}:
                    fp += len(dl.keys())  # TODO: change to add +1 to FP for each falsely detected logo
                fn += 1
                fn_images_list.append(os.path.join(directory_in_str, filename))'''

            # compute confusion matrixes info
            if dl == {}:
                # FN
                local_cms[logo_to_detect][3] += 1
            else:
                if logo_to_detect == max(dl, key=dl.get):
                    # TP
                    local_cms[logo_to_detect][0] += 1
                for name in dl.keys():
                    if not name == logo_to_detect:
                        local_cms[name][2] += 1
            for name in get_logo_names():
                if not name == logo_to_detect and not name in dl.keys():
                    local_cms[name][1] += 1

        if should_print_info:
            print('FN for "', logo_to_detect, '": ', local_cms[logo_to_detect][3])
            print('TP for "', logo_to_detect, '": ', local_cms[logo_to_detect][0])
            print('Could not detect logos in files:')
            for f in fn_images_list:
                print(Fore.RED, ' - ', f)

    return local_cms


def init_confusion_matrices():
    cms = {}
    names = get_logo_names()
    for name in names:
        # Confusion matrix for each logo as list: [TP, TN, FP, FN]
        cms[name] = [0, 0, 0, 0]
    return cms


def update_cms(cms: dict, infos: dict) -> dict:  # TODO: change infos parameter name
    new_cms = cms.copy()
    for name in new_cms.keys():
        new_cms[name] = (cms[name][0] + infos[name][0],
                         cms[name][1] + infos[name][1],
                         cms[name][2] + infos[name][2],
                         cms[name][3] + infos[name][3])
    return new_cms


def update_tresholds(tresholds: dict, cm: dict) -> dict:
    new_tresholds = tresholds.copy()
    all_imgs_nr = sum([sum(cm[name]) for name in new_tresholds.keys()])
    for name in new_tresholds.keys():
        logo_imgs_nr = sum(cm[name])  # logo_imgs_nr = 35
        # all_imgs_nr = 400  # 35
        fp_rate = cm[name][2] / (all_imgs_nr - logo_imgs_nr)
        fn_rate = cm[name][3] / logo_imgs_nr

        # optimization criterion
        if fn_rate < fp_rate:
            new_tresholds[name] += 1
    return new_tresholds


def optimize():
    # TODO: adjust to optimize tresholds for each method (maybe separate file for each method and  just pass the treshold_file names as an attribute)
    print(Fore.BLUE, ' # Optimization started !!!')
    tresholds = load_tresholds()
    new_tresholds = {}
    subfolders_names = ['biedronka', 'zabka']  # [f.name for f in os.scandir('data\\all_pics') if f.is_dir()]

    while True: # not tresholds == new_tresholds:
        # if new_tresholds == {}:
        #     new_tresholds = tresholds
        cms = init_confusion_matrices()

        for name in subfolders_names:
            cm_info = test_all_from_dir('data\\all_pics\\' + name, tresholds)  # TODO: use os
            cms = update_cms(cms, cm_info)

        new_tresholds = update_tresholds(tresholds, cms)
        print('Confusion Matrices:', cms)
        print('New tresholds:', new_tresholds)
        if new_tresholds == tresholds:
            break
        else:
            tresholds = new_tresholds

    print(Fore.BLUE, ' # Optimization finished !!!')
    print('Optimized tresholds:', new_tresholds)
    save_tresholds(new_tresholds)


# Starbucks
'''
dl = detect_logo('adidas_test.png',
                 *get_logo_names(),
                 detection_method=KPD.SURF,
                 matching_method=KPM.FLANN,
                 match_threshold=20,
                 show_match=True,
                 show_detection=False,
                 in_grayscale=False)

dl = detect_logo(
    'danone7.jpg',
    # 'supreme19.jpg',
    *get_logo_names(),
    detection_method=KPD.ORB,
    matching_method=KPM.FLANN,
    match_threshold=45,
    show_match=False,
    show_detection=True,
    in_grayscale=False)

dl = sorted(dl.items(), key=lambda val: val[1])  # , reverse=True)

print("Detected logos:")
print(dl)

'''
# %%
#detect_logo_with_sift('adidas_test.png', 'adidas', show_match=True)


# dl = detect_logo('android12.jpg',
#                 #    *get_logo_names(),
#                  'android',
#                  detection_method=KPD.SIFT,
#                  matching_method=KPM.FLANN,
#                  match_threshold=30,
#                  show_match=False,
#                  show_detection=False,
#                  in_grayscale=False)

# dl = sorted(dl.items(), key=lambda val: val[1])
# print("Detected logos:")
# pprint(dl)

# run_in_loop(detection_method=KPD.SIFT, matching_method=KPM.FLANN,
#             match_threshold=50, show_match=False, show_detection=True)

#run_in_loop(detection_method='SIFT', matching_method='FLANN', match_threshold=50, show_match=False, show_detection=True)



optimize()
