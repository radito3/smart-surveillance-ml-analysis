from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer

class SimplePresenceClassifier(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)

    def get_name(self) -> str:
        return 'simple-presence-classifier-app'

    def process_message(self, yolo_results: list[tuple[any, any, any]]):
        if len(yolo_results) > 0:
            self.publish('classification_results', True)

    def cleanup(self):
        self.publish('classification_results', None)
