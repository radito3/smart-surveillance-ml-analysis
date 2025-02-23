import cv2.typing
import torch
from ultralytics import YOLO

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer
from util.device import get_device

# @software{yolo11_ultralytics,
#           author = {Glenn Jocher and Jing Qiu},
#           title = {Ultralytics YOLO11},
#           version = {11.0.0},
#           year = {2024},
#           url = {https://github.com/ultralytics/ultralytics},
#           orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#           license = {AGPL-3.0}
# }
class ObjectDetector(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        Consumer.__init__(self, broker, 'video_source')
        self.model = None

    def get_name(self) -> str:
        return 'object-detection-app'

    # split the initialization of the model in a separate method, so it can be called from within the worker thread
    # instead of the main thread
    def init(self):
        self.model = YOLO('yolo11m.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def process_message(self, frame: cv2.typing.MatLike):
        with torch.no_grad():
            results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.cpu()
        if len(boxes.data) != 0:
            # class 0 is 'person'
            self.produce_value('object_detection_results', [(bbox, cls) for bbox, cls in zip(boxes.xyxy, boxes.cls) if cls != 0])
        else:
            self.produce_value('object_detection_results', [])

    def cleanup(self):
        self.produce_value('object_detection_results', None)
