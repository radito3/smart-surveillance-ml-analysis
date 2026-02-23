import numpy as np

from messaging.consumer import Consumer
from messaging.message_broker import MessageBroker


class TrainingSink(Consumer):
    def __init__(self, broker: MessageBroker):
        super().__init__(broker)
        self.predictions: list[float] = []

    def accept(self, topic: str, probability: float):
        self.predictions.append(probability)

    def get_predicted_mean(self) -> float:
        return np.mean(self.predictions).__float__()
