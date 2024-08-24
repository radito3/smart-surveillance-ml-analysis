from analysis.types import AnalysisType
from classification.classifier import Classifier


class SuspiciousActivityClassifier(Classifier):

    def __init__(self, suspicious_activities_indices: list[int]):
        # mark specific activities as suspicious
        self.suspicious_activities_indices = suspicious_activities_indices

    def classify_as_suspicious(self, dtype: AnalysisType,  vector: list[any]) -> bool:
        if len(vector) == 0:
            return False

        for idx in self.suspicious_activities_indices:
            if len(vector) > idx and vector[idx] == 1:
                return True
        return False
