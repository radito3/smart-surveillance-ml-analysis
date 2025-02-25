from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from notifications.notification_delegate import send_notification


class BinaryResultSink(Consumer):

    def __init__(self, notification_webhook: str):
        self.notification_webhook_url: str = notification_webhook

    def get_name(self) -> str:
        return 'classification-result-consumer'

    def process_message(self, result: bool):
        if result:
            send_notification(self.notification_webhook_url)
