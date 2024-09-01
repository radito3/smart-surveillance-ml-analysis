from abc import ABC, abstractmethod
from analysis.types import AnalysisType


class BaseAnalyzer(ABC):

    @abstractmethod
    def analysis_type(self) -> AnalysisType:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, payload: any, *args, **kwargs) -> list[any]:
        raise NotImplementedError

    def __call__(self, payload: any, *args, **kwargs) -> list[any]:
        return self.analyze(payload, *args, **kwargs)
