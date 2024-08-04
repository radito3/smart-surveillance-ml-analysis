import cv2

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType
from object_detection.detector import ObjectDetector


class HumanObjectInteractionAnalyzer(SingleFrameAnalyzer):

    def __init__(self, detector: ObjectDetector):
        self.detector = detector

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.HumanObjectInteraction

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        results = self.detector.detect(frame)
        results.summary()
        # TODO: impl
        # instead of having a dedicated model for this, a simpler and more computationally efficient method would be
        # to take the object detection output from the YOLO model and measure the distance and/or border overlap
        # between people and dangerous (predefined for particular scenes) items/objects
        return []
