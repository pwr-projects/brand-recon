from src import *
from src.utils.images import *
from ..core.detector import Detector
from ..core.matcher import Matcher


def get_threshold(method: TM, logo: str, keypoints) -> int:
    threshold = 0
    if method == TM.OPTIMIZED:
        thresholds = load_tresholds()
        threshold = thresholds[logo]
    elif method == TM.PROGRESSIVE:
        threshold = len(keypoints) * 0.03
    elif method == TM.CONSTANT:
        threshold = 10
    return threshold


class Evaluator:
    def __init__(self, d: Detector, m: Matcher, in_grayscale: bool, preproc_mode=PREPROC.NONE, infos=True):
        self._detector = d
        self._matcher = m
        self._in_grayscale = in_grayscale
        self._infos=infos
        self._preproc_mode = preproc_mode

    def predict(self, logos: List[str], photos: List[str], show=False, thresholding_method=TM.OPTIMIZED) -> Mapping[str, str]:
        photos_predictions = dict.fromkeys(photos)
        logos_features = self._get_logos_features(logos)

        if thresholding_method == TM.OPTIMIZED:
            thresholds = load_tresholds()
            if self._infos:
                print(thresholds)

        for photo in photos:
            if self._infos:
                print(" --- Photo:", photo)
            try:
                photo_path = imgpath(photo)
                photo_img = img_load(photo_path, self._in_grayscale)
                if self._preproc_mode == PREPROC.BINARIZATION:
                    photo_img = cv2.adaptiveThreshold(photo_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                elif self._preproc_mode == PREPROC.HIST_EQ:
                    photo_img = cv2.equalizeHist(photo_img)
                photo_features = self._detector(photo_img)
                detected_logos = {}
                for logo in logos:
                    threshold = get_threshold(thresholding_method, logo, logos_features[logo].keypoints)
                    # print(" - Logo:", logo, "with threshold:", threshold)
                    matcheses = self._matcher(logos_features[logo].descriptors, photo_features.descriptors, threshold)
                    if len(matcheses) > 0:
                        # Take the best logo
                        detected_logos[logo] = max([len(matches) for matches in matcheses])

                        if show:
                            test_photo_copy = photo_img.copy()
                            logo_photo = get_logo_photo_by_name(logo, self._in_grayscale)
                            test_photo_copy = show_matched_logo(matcheses[0],
                                                                logos_features[logo].keypoints,
                                                                logo_photo,
                                                                logo,
                                                                photo_features.keypoints,
                                                                test_photo_copy,
                                                                True,
                                                                True)
                            img_show(test_photo_copy)

                if 0 < len(detected_logos.keys()):
                    photos_predictions[photo] = max(detected_logos, key=detected_logos.get)
                else:
                    photos_predictions[photo] = 'Any'
                if self._infos:
                    print(" ---- Detection is: ", photos_predictions[photo], "\n")
            except Exception:
                photos_predictions[photo] = 'Exception'
                print(traceback.format_exc())

        return photos_predictions

    def _get_logos_features(self, logos: List[str]) -> Mapping[str, Features]:
        logos_features = dict.fromkeys(logos)
        for logo in logos:
            if self._infos:
                print(" - Getting features of:", logo)
            logo_photo = get_logo_photo_by_name(logo, self._in_grayscale)
            if self._preproc_mode == PREPROC.BINARIZATION:
                logo_photo = cv2.adaptiveThreshold(logo_photo,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            elif self._preproc_mode == PREPROC.HIST_EQ:
                logo_photo = cv2.equalizeHist(logo_photo)
            logos_features[logo] = self._detector(logo_photo)
        return logos_features

    def eval(self, logos, predictions: Mapping[str, str], annotations):
        true_labels = self._get_true_labels_for_predictions(list(predictions.keys()), annotations)
        hl = self._compute_hamming_loss(list(predictions.values()), list(true_labels.values()))
        cms = self._compute_confusion_matrices(logos, predictions, true_labels)
        if len(logos) > 1:
            rates = self._compute_rates(cms)
        else:
            rates = False
        return hl, cms, rates

    @staticmethod
    def _get_true_labels_for_predictions(photos: List[str], annotations) -> Mapping[str, str]:
        return {photo: annotations[photo] for photo in photos}

    def _compute_hamming_loss(self, predictions, true_logos):
        predictions, true_logos = self._map_labels(predictions, true_logos)
        hl = hamming_loss(true_logos, predictions)
        return hl

    @staticmethod
    def _map_labels(predictions, true_logos):
        mydict = {}
        i = 0
        for item in predictions:
            if i > 0 and item in mydict:
                continue
            else:
                i = i + 1
                mydict[item] = i
        for item in true_logos:
            if i > 0 and item in mydict:
                continue
            else:
                i = i + 1
                mydict[item] = i

        mapped_preds = []
        for item in predictions:
            mapped_preds.append(mydict[item])

        mapped_true_labels = []
        for item in true_logos:
            mapped_true_labels.append(mydict[item])

        return mapped_preds, mapped_true_labels

    @staticmethod
    def _compute_confusion_matrices(logos, predictions, true_labels):
        cms = dict.fromkeys(logos)
        preds = list(predictions.values())
        trues = list(true_labels.values())
        for logo in logos:
            tp = sum([0]+[1 for i in range(len(preds)) if preds[i] == logo and trues[i] == logo])
            tn = sum([0]+[1 for i in range(len(preds)) if not preds[i] == logo and not trues[i] == logo])
            fp = sum([0]+[1 for i in range(len(preds)) if preds[i] == logo and not trues[i] == logo])
            fn = sum([0]+[1 for i in range(len(preds)) if not preds[i] == logo and trues[i] == logo])

            cms[logo] = {'TP': tp,
                         'TN': tn,
                         'FP': fp,
                         'FN': fn}
        return cms

    @staticmethod
    def _compute_rates(cms):
        logo_rates = dict.fromkeys(list(cms.keys()))
        for logo in logo_rates.keys():
            fn_rate = cms[logo]['FN'] / (cms[logo]['FN'] + cms[logo]['TP'])
            fp_rate = cms[logo]['FP'] / (cms[logo]['FP'] + cms[logo]['TN'])
            logo_rates[logo] = {'FNR': fn_rate, 'FPR': fp_rate}
        return logo_rates
