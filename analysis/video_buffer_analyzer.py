import cv2
from abc import abstractmethod
from datetime import timedelta
from .analyzer import BaseAnalyzer


class VideoBufferAnalyzer(BaseAnalyzer):

    # TODO: should the state handling (buffering of frames in this case) be handled here?
    def __init__(self, fps: int, window_size: timedelta):
        self._num_frames: int = int(fps * window_size.total_seconds())

    @abstractmethod
    def analyze(self, video_window: list[cv2.typing.MatLike], *args, **kwargs) -> list[int]:
        pass

    def get_num_frames(self) -> int:
        return self._num_frames
