import os

import cv2.typing
import ultralytics.engine.results
from ultralytics.engine.model import Model
from ultralytics import YOLO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType


class ObjectDetector(SingleFrameAnalyzer):

    def __init__(self):
        self.model: ultralytics.engine.model.Model = YOLO(os.environ["YOLO_MODEL"])

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PersonDetection

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        results = self.detect(frame, True)
        # boxes (Boxes, optional): Object containing detection bounding boxes.
        boxes: ultralytics.engine.results.Boxes = results.boxes.cpu()
        # probs (Probs, optional): Object containing class probabilities for classification tasks.
        probabilities: ultralytics.engine.results.Probs = results.probs.cpu()
        # keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        # keypoints = results.keypoints.cpu()  # is  this needed?
        # class_names_dict = results.names

        return [len(boxes.data), *boxes.xyxy, *boxes.cls, *boxes.conf, *probabilities.data]

    def detect(self, frame: cv2.typing.MatLike, people_only: bool = False) -> ultralytics.engine.results.Results:
        if people_only:
            classes = [0]  # class 0 is 'person'
        else:
            classes = None
        return self.model.predict(frame,
                                  verbose=False,
                                  classes=classes,
                                  conf=0.5  # confidence cut-off threshold
                                  )[0]
