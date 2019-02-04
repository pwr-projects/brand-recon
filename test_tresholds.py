#!/bin/python
# %%
from datetime import datetime

from src import *
from src.core import detector, matcher


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


kpds = [KPD.SIFT]

d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03, sigma=2.5)
m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.72, print_info=False)
evaluator = evaluator.Evaluator(d, m, False, False)

logos = ['biedronka', 'orlen', 'pwr', 'tymbark', 'pip', 'pko', 'cobi', 'pis', "fakt", "tyskie"]
photos = os.listdir(PATH_PHOTOS)
annotations = get_annotations()

print(str(datetime.now()), " # STARTED TESTING THRESHOLDING METHODS")


preds = evaluator.predict(logos, photos, False, thresholding_method=TM.CONSTANT)
hd, cms, rates = evaluator.eval(logos, preds, annotations)
print(str(datetime.now()), " # Test for method", TM.CONSTANT, "finished - Hamming dist:", hd)


preds = evaluator.predict(logos, photos, False, thresholding_method=TM.PROGRESSIVE)
hd, cms, rates = evaluator.eval(logos, preds, annotations)
print(str(datetime.now()), " # Test for method", TM.PROGRESSIVE, "finished - Hamming dist:", hd)


splits = {logo:chunk_it(images.get_photos_with_logo(logo, annotations), 5) for logo in logos}
hds = []

for i in range(0, 5):
    test_photos = [item for sublist in [splits[logo][i] for logo in logos] for item in sublist]
    train_photos = list(set(photos) - set(test_photos))

    reset_thresholds(logos, 20)
    optimizer = Optimizer(False)
    optimizer.optimize(logos, train_photos)

    preds = evaluator.predict(logos, test_photos, False, thresholding_method=TM.OPTIMIZED)
    hd, cms, rates = evaluator.eval(logos, preds, annotations)

    hds.append(hd)
print(str(datetime.now()), " # Test for method", TM.OPTIMIZED, "finished - Hamming dist:", np.mean(hds))

print(str(datetime.now()), " # FINISHED TESTING THRESHOLDING METHODS")
