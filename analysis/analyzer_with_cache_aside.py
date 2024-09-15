from queue import Queue

from .analyzer import BaseAnalyzer
from .types import AnalysisType


class CacheAsideAnalyzer(BaseAnalyzer):

    def __init__(self, cache_queue: Queue, dtype: AnalysisType, delegate: BaseAnalyzer = None):
        self._delegate = delegate
        self._dtype = dtype
        self._cache_queue = cache_queue

    def analysis_type(self) -> AnalysisType:
        return self._dtype

    def analyze(self, payload: any, *args, **kwargs) -> list[any]:
        if self._delegate is not None:
            result = self._delegate.analyze(payload, *args, **kwargs)
            self._cache_queue.put(result)
            return result
        return self._cache_queue.get()
