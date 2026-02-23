from messaging.consumer import Consumer
from messaging.message_broker import MessageBroker
from notifications.notification_delegate import send_notification


class NotificationSink(Consumer):
    def __init__(self, broker: MessageBroker, notification_service_url: str):
        super().__init__(broker)
        self.notification_service_url = notification_service_url

    def accept(self, topic: str, message: any):
        send_notification(self.notification_service_url)
