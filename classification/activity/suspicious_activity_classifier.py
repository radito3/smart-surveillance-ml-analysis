from analysis.types import AnalysisType
from classification.classifier import Classifier


class SuspiciousActivityClassifier(Classifier):

    def __init__(self, suspicious_activities_indices: list[int]):
        # mark specific activities as suspicious
        self.suspicious_activities_indices = suspicious_activities_indices

    def classify_as_suspicious(self, dtype: AnalysisType,  vector: list[any]) -> float:
        if dtype != AnalysisType.ActivityDetection or len(vector) == 0:
            return 0

        for idx in self.suspicious_activities_indices:
            if vector[0] == idx:
                return 1
        return 0
