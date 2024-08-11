from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def classify_as_suspicious(self, vector: list[any]) -> bool:
        pass
