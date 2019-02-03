#!/bin/python
# %%

from datetime import datetime

from src import *
from src.core import detector, matcher

kpds = [KPD.SIFT]

d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03, sigma=2.5)
m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.72, print_info=False)
evaluator = evaluator.Evaluator(d, m, in_grayscale=False, infos=False)

logos = ['biedronka', 'orlen', 'pwr', 'tymbark', 'pip', 'pko', 'cobi', 'pis', "fakt", "tyskie"]
photos = [p for p in os.listdir(PATH_PHOTOS) if 'cobi' in p]
annotations = get_annotations()

print(str(datetime.now()), " # STARTED TESTING BEST CONFIGURATION")

preds = evaluator.predict(logos, photos, False)
hd, cms, rates = evaluator.eval(logos, preds, annotations)

print(str(datetime.now()), " # Test for best configuration with detection method", KPD.SIFT, "finished\n")
print(str(datetime.now()), " # Computed Hamming distance:", hd, "\n")
print(str(datetime.now()), " # Computed  Confusion matrices and TP/TN rates:\n")
for logo in rates.keys():
    print("   -", logo, ": ", rates[logo], " | ", cms[logo])

print(str(datetime.now()), " # FINISHED TESTING BEST CONFIGURATION")
