import grpc
from notifications import notification_service_pb2
from notifications import notification_service_pb2_grpc


# python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. notification_service.proto
def send_notification(payload: any) -> None:
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = notification_service_pb2_grpc.NotificationDelegateServiceStub(channel)
        request = notification_service_pb2.HelloRequest(name='World')
        response: notification_service_pb2.HelloResponse = stub.SendNotification(request)
        print(f"Response from server: {response.message}")
