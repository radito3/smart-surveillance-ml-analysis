import os

import cv2
from rtmlib import RTMO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType


class PoseDetector(SingleFrameAnalyzer):

    def __init__(self):
        # TODO: maybe detect face and hand features separately?
        self.model = RTMO(os.environ["RTMO_MODEL_URL"])

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PoseEstimation

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        # top-level dimension (keypoints/scores[N]) - person N
        # 2nd level (keypoints/scores[N][i]) - keypoints [x, y], scores <float from 0 to 1>
        keypoints, scores = self.model(frame)
        # TODO: filter out keypoints with a lower than 50% score
        return []
