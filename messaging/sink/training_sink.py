import numpy as np

from messaging.consumer import Consumer


class TrainingSink(Consumer):
    def __init__(self):
        self.predictions: list[float] = []

    def get_name(self) -> str:
        return 'training-sink-consumer'

    def get_predicted_mean(self) -> float:
        return np.mean(self.predictions).__float__()

    def process_message(self, probability: float):
        if probability != 0:
            self.predictions.append(probability)
