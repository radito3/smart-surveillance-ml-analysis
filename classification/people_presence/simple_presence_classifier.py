from analysis.types import AnalysisType
from classification.classifier import Classifier


class SimplePresenceClassifier(Classifier):

    def classify_as_suspicious(self, dtype: AnalysisType, vector: list[any]) -> float:
        # if there are any people when there shouldn't, raise an alarm
        return 1 if dtype == AnalysisType.PersonDetection and len(vector) > 0 else 0
