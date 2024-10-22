from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer

class SimplePresenceClassifier(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        Consumer.__init__(self, broker, 'pose_detection_results')

    def get_name(self) -> str:
        return 'simple-presence-classifier-app'

    def consume_message(self, yolo_results: list[tuple[any, any, any]]):
        if len(yolo_results) > 0:
            self.produce_value('classification_results', True)

    def cleanup(self):
        self.produce_value('classification_results', None)
