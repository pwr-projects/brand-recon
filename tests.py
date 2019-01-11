from itertools import product
from run import detect_logo
from src import *

not_working = []
for kpdetector, kpmatcher in product(KPD, KPM):
    # try:
    dl = detect_logo('4363332795.jpg', 'bmw',
                    detection_method=kpdetector,
                    matching_method=kpmatcher,
                    match_threshold=20,
                    show_match=False)
    print("Detected logos:", dl)
    # except Exception as e:
    #     print(e)
    #     not_working.append((kpdetector, kpmatcher))
print('Doesn\'t work', not_working)
