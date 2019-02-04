#!/bin/python
# %%
from datetime import datetime

from src import *
from src.core import detector, matcher
from src.core.evaluator import Evaluator

kpds = [KPD.SIFT]

d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03, sigma=2.5)
m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.72, print_info=False)

logos = ['biedronka', 'orlen', 'pwr', 'tymbark', 'pip', 'pko', 'cobi', 'pis', "fakt", "tyskie"]
photos = os.listdir(PATH_PHOTOS)
annotations = get_annotations()

print(str(datetime.now()), " # STARTED TESTING PREPROCESSING METHODS")

for preprocessing_method in PREPROCESSING_METHODS:
    evaluator = Evaluator(d, m, in_grayscale=True, preproc_mode=preprocessing_method, infos=False)

    preds = evaluator.predict(logos, photos, False)
    hd, cms, rates = evaluator.eval(logos, preds, annotations)
    print(str(datetime.now()), " # Test for method", preprocessing_method, "finished - Hamming dist:", hd)

print(str(datetime.now()), " # FINISHED TESTING PREPROCESSING METHODS")
