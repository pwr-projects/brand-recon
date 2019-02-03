#!/bin/python
# %%

from src import *

kpds = [KPD.SIFT] # Detektor

d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03, sigma=2.5)
m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.75)
eval = Evaluator(d, m, False)

annotations = get_annotations()
logos = ['biedronka']  # Dowolna liczba logo
photos = get_photos_with_logo('biedronka', annotations)  # Dowolne obrazy

preds = eval.predict(logos, photos, True)  # Predykcja logo

# Ewaluacja wynikow
hd, cms, rates = eval.eval(logos, preds, annotations)
print("\n - Hamming dist:", hd)  # Dystans hamminga miedzy wektorem prawdziwych klas a wektorem wynikowym predykcji kalas
print("\n - Conf. matrices:", cms)  # Macierze pomylek dla kazdego logo
if rates:
    print("\n - Rates:")  # Miary: False Negative Rate i False Positive Rate dla kazdego logo
    for logo in rates.keys():
        print("   -", logo, ": ", rates[logo])