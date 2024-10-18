from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from notifications.notification_delegate import send_notification


class ClassificationResultConsumer(Consumer):

    def __init__(self, broker: Broker, notification_webhook: str):
        super().__init__(broker, 'classification_results')
        self.notification_webhook_url: str = notification_webhook

    def get_name(self) -> str:
        return 'classification-result-consumer'

    def consume_message(self, confidence: float):
        if confidence > 0.6:  # experiment with threshold values
            send_notification(self.notification_webhook_url)
