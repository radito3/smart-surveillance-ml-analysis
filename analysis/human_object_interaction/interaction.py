import cv2
import numpy as np

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
        # integrate with YOLO-pose results
        return []

    # Helper function to calculate the Euclidean distance between two points
    @staticmethod
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Check if a person is interacting with an object based on hand keypoints proximity
    # since the people sizes in frames can vary greatly, this distance threshold might not be sufficient
    # try to judge the distance qualifier based on some proportion of the person?
    def is_interacting(self, keypoints, object_box, distance_threshold=50):
        # YOLO-Pose keypoints: [x1, y1] = right hand, [x2, y2] = left hand
        right_hand = keypoints['right_hand']
        left_hand = keypoints['left_hand']

        # Calculate the center of the object's bounding box
        obj_center = [(object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2]

        # Check if either hand is near the object
        return (self.calculate_distance(right_hand, obj_center) < distance_threshold
                or self.calculate_distance(left_hand, obj_center) < distance_threshold)
