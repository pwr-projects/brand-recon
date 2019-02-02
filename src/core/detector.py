from itertools import chain

from ..features import *


class Detector:
    def __init__(self, *kpds: KPD, nOctLay=4, nOct=4, hesThresh=100, ext=True, nfeat=1000, eThresh=10, cThresh=0.03,
                 sigma=1.6):
        assert(all([kpd in (KPD.SIFT, KPD.SURF) for kpd in kpds]) or all(
            [kpd in (KPD.ORB, KPD.BRISK) for kpd in kpds]),
            'Pass subset of {SIFT, SURF}, {ORB, BRISK}')
        self._detector_methods = list(map(self._detector_method, kpds))
        self._brisk = None
        self._surf = None
        self._orb = None
        self._sift = None
        self._nOctLay = nOctLay
        self._nOct = nOct
        self._hesThresh = hesThresh
        self._ext = ext
        self._nfeat = nfeat
        self._eThresh = eThresh
        self._cThresh = cThresh
        self._sigma = sigma

    def __call__(self, photo):
        features = [method(photo) for method in self._detector_methods]
        features = [Features(kps, descs) for (kps, descs) in chain(features)]
        return self._merge_keypoints(*features)

    def _detector_method(self, kpd):
        method_funcs = {KPD.SIFT: self._detect_features_using_sift,
                        KPD.ORB: self._detect_features_using_orb,
                        KPD.SURF: self._detect_features_using_surf,
                        KPD.BRISK: self._detect_features_using_brisk}
        return method_funcs[kpd]

    def _detect_features_using_brisk(self, photo):
        if not self._brisk:
            self._brisk = cv2.BRISK_create()#thresh=30, octaves=5, patternScale=1.0)
        return self._brisk.detectAndCompute(photo, None)

    def _detect_features_using_orb(self, photo):
        if not self._orb:
            self._orb = cv2.ORB_create(  # nfeatures=1500,
                edgeThreshold=20,
                scaleFactor=1.6,
                nlevels=8)
        return self._orb.detectAndCompute(photo, None)

    def _detect_features_using_surf(self, photo):
        if not self._surf:
            self._surf = cv2.xfeatures2d_SURF.create(hessianThreshold=self._hesThresh,
                                                     nOctaves=self._nOct,
                                                     nOctaveLayers=self._nOctLay,
                                                     extended=self._ext)
            return self._surf.detectAndCompute(photo, None)

    def _detect_features_using_sift(self, photo):
        if not self._sift:
            self._sift = cv2.xfeatures2d_SIFT.create(
                nfeatures=self._nfeat,
                nOctaveLayers=self._nOctLay,
                edgeThreshold=self._eThresh,
                contrastThreshold=self._cThresh,
                sigma=self._sigma)
        return self._sift.detectAndCompute(photo, None)

    def _merge_keypoints(self, *features):
        kps = np.asarray(list(chain(*[feat.keypoints for feat in features])))
        descs = [feature.descriptors for feature in features]
        max_width = np.max(list(map(lambda x: x.shape[1], descs)))
        descs = np.array([np.pad(x, (0, max_width - len(x)), 'constant') for x in chain(*descs)])
        return Features(kps, descs)
