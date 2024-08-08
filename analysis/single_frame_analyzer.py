import cv2
from abc import abstractmethod
from .analyzer import BaseAnalyzer


class SingleFrameAnalyzer(BaseAnalyzer):

    @abstractmethod
    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        pass
