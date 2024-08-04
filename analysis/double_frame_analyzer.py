import cv2
from abc import abstractmethod
from .analyzer import BaseAnalyzer


class DoubleFrameAnalyzer(BaseAnalyzer):

    def __init__(self):
        self._pref_frame: cv2.typing.MatLike = None

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        if self._pref_frame is None:
            self._pref_frame = frame
            return []
        result = self.analyze_with_previous_frame(self._pref_frame, frame)
        self._pref_frame = frame
        return result

    @abstractmethod
    def analyze_with_previous_frame(self, frame1: cv2.typing.MatLike, frame2: cv2.typing.MatLike) -> list[int]:
        pass
