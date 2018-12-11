# %%
import cv2
import numpy as np

from src import *


# %%


def detect_logo(photo_name: str,
                *logo_names,
                detection_method: str = 'SIFT',
                matching_method: str = 'FLANN',
                match_threshold: int = 50,
                show_match: bool = True):
    """ Function detecting logos in given photo.

    Args:
        photo_name:       Path to the input photo where the logos are gona be looked for. Example: 'adidas_shoe.png'
        logo_names:       List of logos to look for.
        detection_method: Method of feature extraction from photo.
        matching_method:  Method of key points matching.
        match_threshold:  Threshold of common key points for determining of specific logo presence.
        show_match:       If there will be displayed output image showing matching.

    Returns:
        detected_logos: List of detected logo names.

    """

    detected_logos = []

    photo_path = os.path.join('photos', photo_name)
    photo = img_load(photo_path)
    photo_keypoints, photo_descriptors = detect_features(detection_method, photo)

    logos = get_logos_with_features(detection_method, *logo_names)
    print("\nFound", len(logos), "logos to compare to.")

    for logo_name, (logo_keypoints, logo_descriptors) in logos:
        print("\nComparing input photo having", len(photo_keypoints), "keypoints, with logo '", logo_name, "' having",
              len(logo_keypoints), "keypoints.")

        good_matches = create_good_matches(matching_method, detection_method, logo_descriptors, photo_descriptors)
        print("\nNumber of good matches between input and logo '", logo_name, "' is equal to", len(good_matches))

        if len(good_matches) > match_threshold:
            print(f'Found logo {logo_name}: {len(good_matches)}/{match_threshold}')
            detected_logos.append(logo_name)

            if show_match:
                show_matched_logo(good_matches, logo_keypoints, logo_name, photo, photo_keypoints)

    return detected_logos


def show_matched_logo(good_matches, logo_keypoints, logo_name, photo, photo_keypoints, show_matches=True, show_detection=False):
    matchesMask = None
    photo_copy = photo.copy()
    pattern = get_logo_by_name(logo_name)
    h, w = pattern.shape[:2]
    src_pts = np.float32([logo_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([photo_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    if M.size != 0:
        dst = cv2.perspectiveTransform(pts, M)
        photo_copy = cv2.polylines(photo_copy, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    else:
        # TODO CHECK WHY findHomography sometimes returns empty M
        print('Cannot create this nice matching rect due to... some unknown yet problem :c')
    if show_matches:
        photo_copy = cv2.drawMatches(pattern,
                                     logo_keypoints,
                                     photo_copy,
                                     photo_keypoints,
                                     good_matches,
                                     None,
                                     matchColor=(0, 255, 0),
                                     matchesMask=matchesMask,
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    if show_detection:
        font = cv2.FONT_HERSHEY_SIMPLEX

        writing_x = int(dst[0][0][0])
        writing_y = int(min(dst[0][0][1], dst[3][0][1])-10)

        bottom_left_corner_of_text = (writing_x, writing_y)
        font_scale = 1
        font_color = (255, 0, 128)
        font_type = 2

        cv2.putText(photo_copy, logo_name,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    font_type)

        photo_copy = cv2.cvtColor(photo_copy, cv2.COLOR_GRAY2RGB)

    img_show(photo_copy)


def create_good_matches(matching_method, detection_method, logo_descriptors, photo_descriptors):
    matches = match_descriptors(matching_method, detection_method, logo_descriptors, photo_descriptors)
    good_matches = []

    if matching_method == 'FLANN':
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    elif matching_method == "BF":
        for m in matches:
            if m.distance < 19:
                good_matches.append(m)

    return good_matches


def match_descriptors(method, detection_method, logo_descriptors, photo_descriptors):
    if method == 'FLANN':
        if detection_method == 'SIFT':
            index_params = dict(algorithm=0, trees=100)
            search_params = dict(checks=5000)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(logo_descriptors, photo_descriptors, k=2)

        elif method == 'ORB':
            index_params = dict(algorithm=6,
                                table_number=15,  # 12
                                key_size=20,  # 20
                                multi_probe_level=2)  # 2
            search_params = dict(checks=500)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(logo_descriptors, photo_descriptors, k=2)
    elif method == "BF":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(logo_descriptors, photo_descriptors)
    else:
        raise ValueError("Method must be from:" + str(MATCHING_METHODS))
    return matches


def get_logos_with_features(method, *logo_names):
    logo_names = logo_names if logo_names else get_logo_names()
    logos = {k: v for k, v in get_logo_features(method).items() if k in logo_names}
    logos = tqdm(logos.items(), desc='Logo', disable=not (len(logo_names) > 1 and VERBOSE))
    return logos


def detect_features(method, photo):
    if method == 'SIFT':
        descriptors, keypoints = detect_features_using_sift(photo)
    elif method == "ORB":
        descriptors, keypoints = detect_features_using_orb(photo)
    else:
        raise ValueError("Method must be from:" + str(METHODS))
    return keypoints, descriptors


def detect_features_using_orb(photo):
    orb = cv2.ORB_create(nfeatures=1500,
                         edgeThreshold=25,
                         scaleFactor=1.2,
                         nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(photo, None)
    return descriptors, keypoints


def detect_features_using_sift(photo):
    sift = cv2.xfeatures2d_SIFT.create(edgeThreshold=50,
                                       sigma=2.0,
                                       nOctaveLayers=6)
    keypoints, descriptors = sift.detectAndCompute(photo, None)
    return descriptors, keypoints


def detect_logo_with_sift(photo_path: str, *logo_names, match_threshold: int = 50, show_match: bool = True):
    detected_logos = []

    sift = cv2.xfeatures2d_SIFT.create(edgeThreshold=50,
                                       sigma=2.0,
                                       nOctaveLayers=6)

    photo = img_load(os.path.join('photos', photo_path))# + '.jpg'))
    photo_keypoints, photo_descriptors = sift.detectAndCompute(photo, None)

    logo_names = logo_names if logo_names else get_logo_names()
    logos = {k: v for k, v in get_logo_features_sift().items() if k in logo_names}

    for logo_name, (logo_keypoints, logo_descriptors) in tqdm(logos.items(),
                                                              desc='Logo',
                                                              disable=not (len(logo_names) > 1 and VERBOSE)):
        good_matches = []

        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=100), dict(checks=5000))
        matches = flann.knnMatch(logo_descriptors, photo_descriptors, k=2)

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # good_matches = sorted(matches, key=lambda x: x[0].distance)
        # good_matches = [x for x, _ in matches]

        if len(good_matches) > match_threshold:
            print(f'Found logo {logo_name}: {len(good_matches)}/{match_threshold}')
            detected_logos.append(logo_name)

            if show_match:
                matchesMask = None
                photo_copy = photo.copy()
                pattern = get_logo_by_name(logo_name)
                h, w = pattern.shape[:2]

                src_pts = np.float32([logo_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([photo_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, 0, 5.0)
                matchesMask = mask.ravel().tolist()

                pts = np.float32([[0, 0],
                                  [0, h-1],
                                  [w-1, h-1],
                                  [w-1, 0]]).reshape(-1, 1, 2)

                if M.size != 0:
                    dst = cv2.perspectiveTransform(pts, M)
                    photo_copy = cv2.polylines(photo_copy, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
                else:
                    # TODO CHECK WHY findHomography sometimes returns empty M
                    print('Cannot create this nice matching rect due to... some unknown yet problem :c')

                photo_copy = cv2.drawMatches(pattern,
                                             logo_keypoints,
                                             photo_copy,
                                             photo_keypoints,
                                             good_matches,
                                             None,
                                             matchColor=(0, 255, 0),
                                             matchesMask=matchesMask,
                                             flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                img_show(photo_copy)

    return detected_logos


# %%
#detect_logo_with_sift('adidas_bad.png', 'adidas', show_match=True)

# Example
dl = detect_logo('adidas2_test.png', 'nike', 'adidas', 'adidas2', detection_method='ORB',
                 matching_method='BF', match_threshold=20, show_match=True)
print("Detected logos:", dl)
