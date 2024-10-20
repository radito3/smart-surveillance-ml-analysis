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
        self.model = YOLO('yolo11m-pose.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def consume_message(self, frame: cv2.typing.MatLike):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model(frame, verbose=False, classes=[0])[0]
        # keypoints data format:
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee
        # 15: Left Ankle 16: Right Ankle
        people = results.keypoints.cpu().xy
        self.produce_value('pose_detection_results', people)

    def cleanup(self):
        self.produce_value('pose_detection_results', None)
