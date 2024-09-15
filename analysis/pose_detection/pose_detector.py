import cv2
import torch
from ultralytics import YOLO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType
from util.device import get_device


class PoseDetector(SingleFrameAnalyzer):

    def __init__(self):
        self.model = YOLO('yolov8m-pose.pt').to(get_device())
        self.model.compile() if torch.cuda.is_available() else None

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PoseEstimation

    @torch.no_grad()
    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame, verbose=False, classes=[0])[0]
        people = results.keypoints.cpu().xy
        # keypoints data format: https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
        return people
