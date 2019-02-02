#!/bin/python
# %%

from src import *

all_photos = os.listdir("data\\photos")
biedronka_photos = [p for p in all_photos if 'biedronka' in p]
print('biedronka_photos: ', len(biedronka_photos))
orlen_photos = [p for p in all_photos if 'orlen' in p]
print('orlen_photos: ', len(orlen_photos))
pwr_photos = [p for p in all_photos if 'pwr' in p]
print('pwr_photos: ', len(pwr_photos))
tymbark_photos = [p for p in all_photos if 'tymbark' in p]
print('tymbark_photos: ', len(tymbark_photos))
pip_photos = [p for p in all_photos if 'pip' in p]
print('pip_photos: ', len(pip_photos))
pko_photos = [p for p in all_photos if 'pko' in p]
print('pko_photos: ', len(pko_photos))
cobi_photos = [p for p in all_photos if 'cobi' in p]
print('cobi_photos: ', len(cobi_photos))
pis_photos = [p for p in all_photos if 'pis' in p]
print('pis_photos: ', len(pis_photos))
fakt_photos = [p for p in all_photos if 'fakt' in p]
print('fakt_photos: ', len(fakt_photos))
tyskie_photos = [p for p in all_photos if 'tyskie' in p]
print('tyskie_photos: ', len(tyskie_photos))


