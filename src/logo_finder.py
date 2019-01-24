from .config import *


class LogoFinder:
    def __init__(self):
        self._logo_features = None

    def __call__(self, *img_paths):
        assert all(map(isphoto, img_paths)), 'Some of passed photo paths do not exist'
