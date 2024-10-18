from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer

class SimplePresenceClassifier(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        Consumer.__init__(self, broker, 'object_detection_results')

    def get_name(self) -> str:
        return 'simple-presence-classifier-app'

    def consume_message(self, yolo_results: list[tuple[any, any, any, any]]):
        detected_people = [True for _, _, cls, _ in yolo_results if cls == 0]
        if len(detected_people) > 0:
            self.produce_value('classification_results', True)

    def cleanup(self):
        self.produce_value('classification_results', None)
