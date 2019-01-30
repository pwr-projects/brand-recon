#!/bin/python
# %%

from src import *

kpds = [KPD.SIFT] #, KPD.SURF]

d = detector.Detector(*kpds, nOctLay=3, nOct=4, hesThresh=100, ext=True, nfeat=10000, eThresh=20, cThresh=0.03, sigma=2.5)
m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=0.73)
eval = evaluator.Evaluator(d, m, False)


all_photos = os.listdir("data\\photos")
zabka_photos = [p for p in all_photos if 'zabka' in p]
biedronka_photos = [p for p in all_photos if 'biedronka' in p]
lot_photos = [p for p in all_photos if 'lot' in p]
orlen_photos = [p for p in all_photos if 'orlen' in p]
pwr_photos = [p for p in all_photos if 'pwr' in p]
tymbark_photos = [p for p in all_photos if 'tymbark' in p]
lotos_photos = [p for p in all_photos if 'lotos' in p]
pip_photos = [p for p in all_photos if 'pip' in p]
doz_photos = [p for p in all_photos if 'doz' in p]
pko_photos = [p for p in all_photos if 'pko' in p]

logos = ['biedronka', 'orlen', 'pwr', 'tymbark', 'lotos', 'pip', 'doz', 'pko']  # ['lot', 'biedronka', 'zabka', 'orlen', 'pwr', 'plus']
photos = pwr_photos+biedronka_photos+orlen_photos+tymbark_photos+pip_photos+lotos_photos+pko_photos+doz_photos # all_photos # (orlen_photos+lot_photos) #  (lot_photos[:10])  # +biedronka_photos[:10])
preds = eval.predict(logos, photos, False)  # ['biedronka_img_2.jpg', 'biedronka_img_14.jpg', 'biedronka_img_17.jpg'])
# print(preds)

hd, cms, rates = eval.eval(logos, preds)
print("\n - Hamming dist:", hd)
print("\n - Conf. matrices:", cms)
print("\n - Rates:")
for logo in rates.keys():
    print("   -", logo, ": ", rates[logo])


'''all_photos = os.listdir("data\\photos")
logos = ['lot', 'biedronka', 'zabka', 'orlen', 'pwr', 'plus']
photos = all_photos

for tresh_factor in [0.015, 0.025, 0.035]:
    for matcher_tresh in [0.5, 0.6, 0.7, 0.8]:
        for hessT in [50, 100, 150]:
            for nOct in [3, 4, 5]:
                for nOctLay in [3, 4, 5]:
                    for ext in [True, False]:
                        for nf in [500, 1500, 5000, 20000]:
                            for eT in [5, 10, 15]:
                                for cT in [0.01, 0.03, 0.09]:
                                    for s in [1.1, 1.6, 2.2]:
                                        print('\n\n\n\n')
                                        print(' ----- Test Configuration ------ ')
                                        print(' - Treshold factor:', tresh_factor)
                                        print(' - Matcher tresh. factor:', matcher_tresh)
                                        print(' - SURF - hessianThreshold:', hessT)
                                        print(' - SURF - nOctaves:', nOct)
                                        print(' - SURF - nOctaveLayers:', nOctLay)
                                        print(' - SURF - extended:', ext)
                                        print(' - SIFT - nfeatures:', nf)
                                        print(' - SIFT - edgeThreshold:', eT)
                                        print(' - SIFT - contrastThreshold:', cT)
                                        print(' - SIFT - sigma:', s)
                                        print('\n')

                                        kpds = [KPD.SIFT, KPD.SURF]

                                        d = detector.Detector(*kpds,
                                                              nOctLay=nOctLay,
                                                              nOct=nOct,
                                                              hesThresh=hessT,
                                                              ext=ext,
                                                              nfeat=nf,
                                                              eThresh=eT,
                                                              cThresh=cT,
                                                              sigma=s)
                                        m = matcher.Matcher(KPM.FLANN, *kpds, tresh_factor=matcher_tresh)
                                        eval = evaluator.Evaluator(d, m, False, tresh_factor)

                                        preds = eval.predict(logos, photos, False)

                                        hd, cms, rates = eval.eval(logos, preds)
                                        print("\n - Hamming dist:", hd)
                                        print("\n - Conf. matrices:", cms)
                                        print("\n - Rates:")
                                        for logo in rates.keys():
                                            print("   -", logo, ": ", rates[logo])

                                        print(' --------- END -----------\n\n\n\n')'''
