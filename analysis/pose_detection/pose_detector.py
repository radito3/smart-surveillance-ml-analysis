import cv2
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
class PoseDetector(Producer, Consumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        self.model = None

    def get_name(self) -> str:
        return 'pose-detection-app'

    def init(self):
        self.model = YOLO('yolo11m-pose.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def process_message(self, frame: cv2.typing.MatLike):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            # we need `track` instead of `predict` because we need to keep track of people between frames
            # persist=True to preserve tracker IDs between calls
            results = self.model.track(frame, persist=True, verbose=False, classes=[0])[0]
        # keypoints data format:
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee
        # 15: Left Ankle 16: Right Ankle
        bboxes = results.boxes.cpu()
        if len(bboxes.data) != 0:
            kpts = results.keypoints.cpu().xy
            ids = bboxes.id
            if ids is None:
                # a non-existent ID so that no entries match it
                ids = [torch.tensor(-2.0) for _ in range(len(kpts))]
            self.publish('pose_detection_results', [*zip(bboxes.xyxy, ids, kpts)])
        else:
            self.publish('pose_detection_results', [])

    def cleanup(self):
        self.publish('pose_detection_results', None)
