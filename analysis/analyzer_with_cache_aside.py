import queue
from queue import Queue

from .analyzer import BaseAnalyzer
from .types import AnalysisType


class CacheAsideAnalyzer(BaseAnalyzer):

    def __init__(self, cache_queue: Queue, dtype: AnalysisType, delegate: BaseAnalyzer = None):
        self._delegate: BaseAnalyzer = delegate
        self._dtype: AnalysisType = dtype
        self._cache_queue: Queue = cache_queue

    def analysis_type(self) -> AnalysisType:
        return self._dtype

    def analyze(self, payload: any, *args, **kwargs) -> list[any]:
        if self._delegate is not None:
            result = self._delegate.analyze(payload, *args, **kwargs)
            try:
                self._cache_queue.put_nowait(result)
            except queue.Full:
                # this exception occurs from time to time
                # when it happens, we will skip a frame in the activity recognizer - is that okay?
                pass
            return result

        return self._cache_queue.get()
