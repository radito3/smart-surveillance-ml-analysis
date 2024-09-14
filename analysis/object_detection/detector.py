import os
import cv2.typing
import torch
import ultralytics.engine.results
from ultralytics.engine.model import Model
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType
from util.device import get_device


class ObjectDetector(SingleFrameAnalyzer):

    def __init__(self):
        self.model = None
        self.device = get_device()

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PersonDetection

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        results = self.detect(frame, True)
        # is it necessary for copying to CPU memory?
        boxes: ultralytics.engine.results.Boxes = results.boxes.cpu()  # bounding boxes
        if len(boxes.data) == 0:
            return []

        return [*zip(boxes.xyxy, boxes.id, boxes.conf)]

    @torch.no_grad()
    def detect(self, frame: cv2.typing.MatLike, people_only: bool = False) -> ultralytics.engine.results.Results:
        if self.model is None:
            # lazy initialization, due to serialization issues
            self.model = YOLO(os.environ["YOLO_MODEL"])
            self.model.to(self.device)

        if people_only:
            classes = [0]  # class 0 is 'person'
        else:
            classes = None

        # for debugging visually:
        # boxes = results[i].boxes.xyxy.cpu()
        # clss = results[i].boxes.cls.cpu().tolist()
        # track_ids = results[i].boxes.id.int().cpu().tolist()
        #
        # annotator = Annotator(frame, line_width=2)
        #
        # for box, cls, track_id in zip(boxes, clss, track_ids):
        #     annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} {track_id}")

        # we need `track` instead of `predict` because we need to keep track of objects between frames
        # this may not be needed if we aren't using the GraphLSTM classifier
        return self.model.track(frame,
                                persist=True,  # track-specific argument
                                verbose=False,
                                classes=classes,
                                conf=0.5  # confidence cut-off threshold
                                )[0]
