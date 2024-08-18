from classification.classifier import Classifier


class SimplePresenceClassifier(Classifier):

    def classify_as_suspicious(self, vector: list[any]) -> bool:
        # if there are any people when there shouldn't, raise an alarm
        return len(vector) > 0