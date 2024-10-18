import cv2
import torch
from ultralytics import YOLO

from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer
from util.device import get_device


class PoseDetector(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        Consumer.__init__(self, broker, 'video_source')
        self.model = None

    def get_name(self) -> str:
        return 'pose-detection-app'

    def init(self):
        self.model = YOLO('yolov8m-pose.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def consume_message(self, frame: cv2.typing.MatLike):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model(frame, verbose=False, classes=[0])[0]
        # keypoints data format: https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        people = results.keypoints.cpu().xy
        if len(people) != 0:
            self.produce_value('pose_detection_results', people)

    def cleanup(self):
        self.produce_value('pose_detection_results', None)
