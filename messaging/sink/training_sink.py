import numpy as np

from messaging.processor import MessageProcessor


class TrainingSink(MessageProcessor):
    def __init__(self):
        super().__init__()
        self.predictions: list[float] = []

    def process(self, probability: float):
        self.predictions.append(probability)

    def get_predicted_mean(self) -> float:
        return np.mean(self.predictions).__float__()
