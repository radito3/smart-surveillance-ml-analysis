
class SimplePresenceClassifier:

    def classify_as_suspicious(self, dtype: any, vector: list[any]) -> float:
        # if there are any people when there shouldn't, raise an alarm
        return 1 if len(vector) > 0 else 0
