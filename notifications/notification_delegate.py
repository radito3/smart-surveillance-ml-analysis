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
            print(f"Sending notification failed: {err}")
            retry_attempts -= 1
    print(f"Could not send notification within {default_retry_attempts} attempts. Consider investigating the issue")


def _send_notification_internal(address: str) -> None:
    # TODO: secure channel with a TLS certificate
    with grpc.insecure_channel(address) as channel:
        stub = notification_service_pb2_grpc.NotificationDelegateServiceStub(channel)
        request = notification_service_pb2.HelloRequest(name='World')
        # TODO: replace with more robust logging
        print(f"Sending notification to: {address}")
        stub.SendNotification(request)
        print("Sent notification")
