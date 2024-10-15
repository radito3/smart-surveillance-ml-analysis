import logging
import time
import grpc
from notifications import notification_service_pb2
from notifications import notification_service_pb2_grpc


# python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. notification_service.proto
def send_notification(address: str) -> None:
    default_retry_attempts: int = 3
    retry_attempts: int = default_retry_attempts
    while retry_attempts > 0:
        try:
            _send_notification_internal(address)
            return
        except Exception as err:
            logging.debug(f"Sending notification failed: {err}. Retrying after 2 seconds")
            retry_attempts -= 1
            time.sleep(2)
    logging.error(f"Could not send notification within {default_retry_attempts} attempts. Consider investigating the issue")


def _send_notification_internal(address: str) -> None:
    # strip the path, which contains the camera ID and use it in the message
    # TODO: secure channel with a TLS certificate
    with grpc.insecure_channel(address) as channel:
        stub = notification_service_pb2_grpc.NotificationDelegateServiceStub(channel)
        # TODO: the payload for the message should be the ID of the camera source sending this notification
        request = notification_service_pb2.HelloRequest(name='World')
        logging.info(f"Sending notification to: {address}")
        stub.SendNotification(request)
        logging.info("Sent notification")
