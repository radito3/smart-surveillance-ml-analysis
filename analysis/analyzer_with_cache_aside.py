from .analyzer import BaseAnalyzer
from .types import AnalysisType


class CacheAsideAnalyzer(BaseAnalyzer):

    def __init__(self, delegate: BaseAnalyzer, cache_life: int):
        self._delegate = delegate
        self._cached_value = None
        self._max_cache_life = cache_life
        self._remaining_cache_life = 0

    def analysis_type(self) -> AnalysisType:
        return self._delegate.analysis_type()

    def analyze(self, payload: any, *args, **kwargs) -> list[any]:
        if self._remaining_cache_life > 0:
            self._remaining_cache_life -= 1
            return self._cached_value
        result = self._delegate.analyze(payload, *args, **kwargs)
        self._cached_value = result
        self._remaining_cache_life = self._max_cache_life
        return result
