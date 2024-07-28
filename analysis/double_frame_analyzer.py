import cv2
from abc import abstractmethod
from .single_frame_analyzer import SingleFrameAnalyzer


class DoubleFrameAnalyzer(SingleFrameAnalyzer):

    @abstractmethod
    def analyze(self, frame1: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        pass

    def analyze_with_previous_frame(self, frame1: cv2.typing.MatLike, frame2: cv2.typing.MatLike) -> list[int]:
        return self.analyze(frame1, frame2)
