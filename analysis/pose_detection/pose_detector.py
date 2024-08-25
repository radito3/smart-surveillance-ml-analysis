import os

import cv2
import torch
from rtmlib import RTMO

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType


class PoseDetector(SingleFrameAnalyzer):

    def __init__(self):
        self.model = None

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PoseEstimation

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[any]:
        if self.model is None:
            # lazy initialization is done because the RTMO model is not serializable and can't be sent to a child process
            self.model = RTMO(os.environ["RTMO_MODEL_URL"], device='cuda' if torch.cuda.is_available() else 'cpu')

        # top-level dimension (keypoints/scores[N]) - person N
        # 2nd level (keypoints/scores[N][i]) - keypoints [x, y], scores <float from 0 to 1>
        keypoints, scores = self.model(frame)

        # keypoints data format: https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md

        # for debugging:
        # frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        # do not filter out low-confidence key points, as confidence will be important for orientation detection
        result = [
            # x, y, confidence
            [(kpt[0], kpt[1], score) for kpt, score in zip(kpts, kscores)]
            for kpts, kscores in zip(keypoints, scores)
        ]

        return result
