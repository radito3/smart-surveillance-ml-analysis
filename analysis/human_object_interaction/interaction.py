import cv2

from analysis.single_frame_analyzer import SingleFrameAnalyzer
from analysis.types import AnalysisType
from analysis.object_detection.detector import ObjectDetector


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

        # a more accurate feature vector would take into consideration the euclidian distance between certain body
        # parts and objects - if it's sufficiently small, then it would be reasonable to assume that the person is
        # directly interacting with that object (e.g. holding a knife)
        # + the amount of time an object has been "held" could hold some valuable info?
        # + if there are rapid changes in position (high velocity) in both the object and the human body part closest
        #   to it - that would probably mean rapid movement of a potentially dangerous object

        # another potentially beneficial piece of information would be if there are objects with high velocities
        # in the frame - that could mean something is thrown, falling, sliding, etc., which could be suspicious
        return []
