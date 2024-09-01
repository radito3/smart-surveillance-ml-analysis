import cv2
from abc import abstractmethod
from datetime import timedelta
from .analyzer import BaseAnalyzer


class VideoBufferAnalyzer(BaseAnalyzer):

    def __init__(self, fps: int, window_size: timedelta, window_step: int):
        self._num_frames: int = int(fps * window_size.total_seconds())
        self._buffer: list[any] = []
        self._window_step = window_step

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        if len(self._buffer) < self._num_frames:
            self._buffer.append(self.buffer_hook(frame))
            return []
        result = self.analyze_video_window(self._buffer)
        self._buffer = self._buffer[self._window_step:]
        return result

    def buffer_hook(self, frame: cv2.typing.MatLike) -> any:
        return frame

    @abstractmethod
    def analyze_video_window(self, window: list[any]) -> list[any]:
        pass
