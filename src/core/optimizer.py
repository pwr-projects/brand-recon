from colorama import Fore

from ..utils import *


class CM:
    def __init__(self, *logo_names):
        self._cms = {name: [0, ] * 4 for name in logo_names}

    def update(self, cms: dict, infos: dict):
        new_cms = cms.copy()
        for name in new_cms.keys():
            new_cms[name] = (self._cms[name][0] + infos[name][0],
                             self._cms[name][1] + infos[name][1],
                             self._cms[name][2] + infos[name][2],
                             self._cms[name][3] + infos[name][3])
        return new_cms


def load_tresholds() -> dict:
    tresh_path = PATH_TRESHOLDS()
    if os.path.isfile(tresh_path):
        with open(tresh_path, 'r') as fhd:
            return json.load(fhd)
    else:
        return {name: 10 for name in get_possible_logo_names()}


def update_tresholds(tresholds: dict, cm: dict) -> dict:
    new_tresholds = tresholds.copy()
    all_imgs_nr = sum([sum(cm[name]) for name in new_tresholds.keys()])

    for name in new_tresholds.keys():
        logo_imgs_nr = sum(cm[name])
        fp_rate = cm[name][2] / (all_imgs_nr - logo_imgs_nr)
        fn_rate = cm[name][3] / logo_imgs_nr

        # optimization criterion
        if fn_rate < fp_rate:
            new_tresholds[name] += 1

    return new_tresholds


class Optimizer:
    def __init__(self, init_value: int = 25):
        self._tresholds = load_tresholds()
        self._init_value = init_value

    def save_tresholds(self):
        with open(PATH_TRESHOLDS, 'w+') as fhd:
            json.dump(self._tresholds, fhd)

    def __call__(self, annotations, *logo_names):
         # TODO: adjust to optimize tresholds for each method (maybe separate file for each method and  just pass the treshold_file names as an attribute)
        print(Fore.BLUE, ' # Optimization started !!!')
        new_tresholds = {}
        all_logo_names = get_possible_logo_names()

        while True:
            cms = CM(*logo_names)

            for name in logo_names:
                cm_info = test_all_from_dir(name, annotations, all_logo_names)
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

    def test_all_from_dir(self, logo_name, annotations, all_logo_names, should_print_info=False):
        fn_images_list = []

        local_cms = init_confusion_matrices()

        for filename in get_photos_with_logo(logo_name, annotations):
            dl = detect_logo(file_path,
                                 *all_logo_names,
                                 tresholds=self._tresholds,
                                 detection_method=KPD.ORB,
                                 matching_method=KPM.FLANN,
                                 match_threshold=35,
                                 show_match=False,
                                 show_detection=False,
                                 in_grayscale=False)

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
