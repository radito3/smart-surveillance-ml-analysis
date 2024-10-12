import os
from analysis.types import AnalysisType
from classification.classifier import Classifier


class SuspiciousActivityClassifier(Classifier):

    def __init__(self):
        env_whitelist = os.environ['ACTIVITY_WHITELIST']
        if env_whitelist is not None and len(env_whitelist) != 0:
            self.whitelist_activities_indices = [int(idx) for idx in env_whitelist]
        else:
            self.whitelist_activities_indices = [2, 8]  # temp

    def classify_as_suspicious(self, dtype: AnalysisType,  vector: list[any]) -> float:
        if dtype != AnalysisType.ActivityDetection or len(vector) == 0:
            return 0

        for activity_idx in vector:
            if activity_idx not in self.whitelist_activities_indices:
                return 1
        return 0
