import os
import re

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from notifications.notification_delegate import send_notification


class ProbabilityResultConsumer(Consumer):

    def __init__(self, broker: Broker, notification_webhook: str):
        super().__init__(broker, 'classification_results')
        self.notification_webhook_url: str = notification_webhook
        if 'SINK_PROBABILITY_THRESHOLD' in os.environ:
            threshold = os.environ['SINK_PROBABILITY_THRESHOLD']
            if bool(re.match(r'^[01]\.\d*$', threshold)):
                self.probability_threshold: float = float(threshold)
        else:
            self.probability_threshold: float = 0.6  # default threshold

    def get_name(self) -> str:
        return 'classification-result-consumer'

    def consume_message(self, confidence: float):
        if confidence > self.probability_threshold:  # experiment with threshold values
            send_notification(self.notification_webhook_url)
