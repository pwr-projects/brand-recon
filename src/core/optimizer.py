from colorama import Fore

from src.core import detector, matcher, Evaluator
from ..utils import *


def update_tresholds(tresholds: dict, rates: dict) -> dict:
    new_tresholds = tresholds.copy()

    for name in new_tresholds.keys():
        # optimization criterion
        if name in rates.keys():
            print(Fore.CYAN, " $$$ ", name, rates[name])
            if rates[name]['FNR'] > 0. and rates[name]['FNR'] > rates[name]['FPR'] * 5 and new_tresholds[name] > 3:
                new_tresholds[name] -= 1

    return new_tresholds


class Optimizer:
    def __init__(self, infos=True):
        self._infos=infos

    def optimize(self, logos, photos):
        print(Fore.BLUE, ' # Optimization started !!!')
        new_tresholds = {}
        kpds = [KPD.SIFT]
        d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03,
                              sigma=2.5)
        m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.72, print_info=self._infos)
        evaluator = Evaluator(d, m, False, infos=self._infos)

        tresholds = load_tresholds()
        while True:
            predictions = evaluator.predict(logos, photos, False)
            _, _, rates = evaluator.eval(logos, predictions, False)

            new_tresholds = update_tresholds(tresholds, rates)
            print(Fore.LIGHTBLUE_EX, 'New tresholds:', new_tresholds)
            if new_tresholds == tresholds:
                break
            else:
                tresholds = new_tresholds
                save_tresholds(new_tresholds)

        print(Fore.BLUE, ' # Optimization finished !!!')
        print(Fore.BLUE, 'Optimized tresholds:', new_tresholds)
        save_tresholds(new_tresholds)
