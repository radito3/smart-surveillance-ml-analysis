import numpy as np

from messaging.broker_interface import Broker
from messaging.consumer import Consumer


class TrainingSink(Consumer):
    def __init__(self, broker: Broker):
        super().__init__(broker, 'classification_results')
        self.predictions: list[float] = []

    def get_name(self) -> str:
        return 'training-sink-consumer'

    def get_predicted_mean(self) -> float:
        return np.mean(self.predictions).__float__()

    def consume_message(self, probability: float):
        if probability != 0:
            self.predictions.append(probability)
