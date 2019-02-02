#!/bin/python
# %%

from src import *

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
pzu_photos = [p for p in all_photos if 'pzu' in p]
cobi_photos = [p for p in all_photos if 'cobi' in p]
sonko_photos = [p for p in all_photos if 'sonko' in p]
mlekovita_photos = [p for p in all_photos if 'mlekovita' in p]
pis_photos = [p for p in all_photos if 'pis' in p]
hm_photos = [p for p in all_photos if 'hm' in p]
po_photos = [p for p in all_photos if 'po' in p]
fakt_photos = [p for p in all_photos if 'fakt' in p]
tyskie_photos = [p for p in all_photos if 'tyskie' in p]

logos = ['biedronka', 'orlen', 'pwr', 'tymbark', 'pip', 'pko', 'cobi', 'pis', "fakt", "tyskie"]  # ['lot', 'biedronka', 'zabka', 'orlen', 'pwr', 'plus']
photos = pwr_photos[:15]+\
         biedronka_photos[:15]+\
         orlen_photos[:15]+\
         tymbark_photos[:15]+\
         pip_photos[:15]+\
         pko_photos[:15]+tyskie_photos[:15]+fakt_photos[:15]+\
         cobi_photos[:15]+pis_photos[:15] # all_photos # (orlen_photos+lot_photos) #  (lot_photos[:10])  # +biedronka_photos[:10]) pzu_photos+\mlekovita_photos+\

optimizer = Optimizer()
optimizer.optimize(logos, photos)