import cv2.typing
import torch
import ultralytics.engine.results
from ultralytics import YOLO

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer
from util.device import get_device


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

    def consume_message(self, frame: cv2.typing.MatLike):
        with torch.no_grad():
            # we need `track` instead of `predict` because we need to keep track of objects between frames
            # this may not be needed if we aren't using the GraphLSTM classifier
            # persist=True to preserve tracker IDs between calls
            results = self.model.track(frame, persist=True, verbose=False)[0]
        boxes: ultralytics.engine.results.Boxes = results.boxes.cpu()  # bounding boxes
        if len(boxes.data) != 0:
            self.produce_value('object_detection_results', [*zip(boxes.xyxy, boxes.id, boxes.cls, boxes.conf)])
        else:
            self.produce_value('object_detection_results', [])

    def cleanup(self):
        self.produce_value('object_detection_results', None)
