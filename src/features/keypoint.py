import cv2

__all__ = ['KeyPoint']


class KeyPoint:
    def __init__(self, keypoint: cv2.KeyPoint):
        self._keypoint_cpp = keypoint

    @classmethod
    def create(cls,
               x: float, y: float,
               size: float,
               angle: float = -1.0,
               response: float = 0.0,
               octave: int = 0,
               class_id: int = -1):
        return KeyPoint(cv2.KeyPoint(x, y,
                                     size, angle,
                                     response, octave, class_id))

    @property
    def pt(self):
        return self._keypoint_cpp.pt

    @property
    def size(self):
        return self._keypoint_cpp.size

    @property
    def angle(self):
        return self._keypoint_cpp.angle

    @property
    def response(self):
        return self._keypoint_cpp.response

    @property
    def octave(self):
        return self._keypoint_cpp.octave

    @property
    def class_id(self):
        return self._keypoint_cpp.class_id

    @property
    def cpp(self):
        return self._keypoint_cpp
