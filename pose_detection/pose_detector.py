import os
from functools import reduce
import operator

import cv2
from rtmlib import RTMO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType


class PoseDetector(SingleFrameAnalyzer):

    def __init__(self):
        # TODO: maybe detect face and hand features separately?
        self.model = None

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PoseEstimation

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        if self.model is None:
            # lazy initialization is done because the RTMO model is not serializable and can't be sent to a child process
            self.model = RTMO(os.environ["RTMO_MODEL_URL"])

        # top-level dimension (keypoints/scores[N]) - person N
        # 2nd level (keypoints/scores[N][i]) - keypoints [x, y], scores <float from 0 to 1>
        keypoints, scores = self.model(frame)
        # filter out low-confidence key points
        result = [
            # kpt is a float32 array/tensor? of two elements: x, y coordinates
            [kpt for kpt, score in zip(kpts, kscores) if score > 0.5]
            for kpts, kscores in zip(keypoints, scores)
        ]

        return [*reduce(operator.concat, result)]
