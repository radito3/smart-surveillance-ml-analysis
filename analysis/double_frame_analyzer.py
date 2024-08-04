import cv2
from abc import abstractmethod
from .single_frame_analyzer import SingleFrameAnalyzer


class DoubleFrameAnalyzer(SingleFrameAnalyzer):

    def __init__(self):
        self._pref_frame: cv2.typing.MatLike = None

    def analyze(self, frame1: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        if self._pref_frame is None:
            self._pref_frame = frame1
            return []
        result = self.analyze(self._pref_frame, frame1)
        self._pref_frame = frame1
        return result

    @abstractmethod
    def analyze_with_previous_frame(self, frame1: cv2.typing.MatLike, frame2: cv2.typing.MatLike) -> list[int]:
        pass

    def get_num_frames(self) -> int:
        return 2
