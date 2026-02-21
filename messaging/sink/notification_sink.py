from messaging.consumer import Consumer
from notifications.notification_delegate import send_notification


class NotificationSink(Consumer):
    def __init__(self, notification_service_url: str):
        self.notification_service_url = notification_service_url

    def process(self, message: any):
        send_notification(self.notification_service_url)
