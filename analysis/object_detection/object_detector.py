import cv2.typing
import torch
from ultralytics import YOLO

from messaging.processor import MessageProcessor
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
class ObjectDetector(MessageProcessor):

    def __init__(self):
        super().__init__()
        self.model = None

    # split the initialization of the model in a separate method, so it can be called from within the worker thread
    # instead of the main thread
    def init(self):
        self.model = YOLO('yolo11m.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def process(self, frame: cv2.typing.MatLike):
        with torch.no_grad():
            results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.cpu()
        # filter out class 0 ('person')
        # divide by the total number of classes from the COCO dataset (80) to normalize the value within the range [0, 1]
        output = [] if len(boxes.data) == 0 else [(bbox, cls / 79) for bbox, cls in zip(boxes.xyxy, boxes.cls) if cls != 0]
        self.next(output)
