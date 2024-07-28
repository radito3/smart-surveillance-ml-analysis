import cv2
from abc import abstractmethod
from .analyzer import BaseAnalyzer


class VideoBufferAnalyzer(BaseAnalyzer):

    @abstractmethod
    def analyze(self, video_window: list[cv2.typing.MatLike], *args, **kwargs) -> list[int]:
        pass
