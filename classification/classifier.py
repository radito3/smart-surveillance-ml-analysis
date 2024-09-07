from abc import ABC, abstractmethod

from analysis.types import AnalysisType


class Classifier(ABC):

    @abstractmethod
    def classify_as_suspicious(self, dtype: AnalysisType, vector: list[any]) -> float:
        raise NotImplementedError
