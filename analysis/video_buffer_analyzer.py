import cv2
from abc import abstractmethod
from datetime import timedelta
from .analyzer import BaseAnalyzer


class VideoBufferAnalyzer(BaseAnalyzer):

    # TODO: how to do a sliding window?
    def __init__(self, fps: int, window_size: timedelta):
        self._num_frames: int = int(fps * window_size.total_seconds())
        self._buffer: list[cv2.typing.MatLike] = []

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        if len(self._buffer) < self._num_frames:
            self._buffer.append(frame)
            return []
        result = self.analyze_video_window(self._buffer)
        self._buffer.clear()
        return result

    @abstractmethod
    def analyze_video_window(self, window: list[cv2.typing.MatLike]) -> list[int]:
        pass
