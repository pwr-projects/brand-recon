from typing import List
from typing import Mapping


class Tester:
    def __init__(self, ):
        pass

    def test(self, logos: List[str], photos: List[str]) -> Mapping[str, str]:
        photos_predictions = dict.fromkeys(photos)
        for photo in photos:
            pass
            # detected_logos
            # for logo in logos:
                # detection = detect(logo, photo)
                # if detection not Null: detected_logos.add(detection)
            # best_logo = max(detected_logos) else "any"
            # photos_predictions[photo] = best_logo
        return photos_predictions

