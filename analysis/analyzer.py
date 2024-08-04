from abc import ABC, abstractmethod
from analysis.types import AnalysisType


class BaseAnalyzer(ABC):

    @abstractmethod
    def analysis_type(self) -> AnalysisType:
        pass

    @abstractmethod
    def analyze(self, payload: any, *args, **kwargs) -> list[int]:
        raise AssertionError("abstract class method called")

    def __call__(self, payload: any, *args, **kwargs) -> list[int]:
        return self.analyze(payload, *args, **kwargs)
