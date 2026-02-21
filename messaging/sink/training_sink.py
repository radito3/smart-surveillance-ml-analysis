import numpy as np

from messaging.consumer import Consumer


class TrainingSink(Consumer):
    def __init__(self):
        self.predictions: list[float] = []

    def process(self, probability: float):
        self.predictions.append(probability)

    def get_predicted_mean(self) -> float:
        return np.mean(self.predictions).__float__()
