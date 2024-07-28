import os

import cv2.typing
import ultralytics.engine.results
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):
        self.model = YOLO(os.environ["YOLO_MODEL"])

    def detect(self, frame: cv2.typing.MatLike) -> ultralytics.engine.results.Results:
        return self.model.predict(frame,
                                  verbose=False,
                                  # classes=[0],  # match only people
                                  conf=0.5
                                  )[0]  # confidence cut-off threshold
