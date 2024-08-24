import os

import cv2
import torch
from rtmlib import RTMO
import numpy as np

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

    def classify_position(self, keypoints):
        # Example keypoints (you may need to adapt this based on the specific model)
        head_y = keypoints[0][1]
        shoulders_y = (keypoints[5][1] + keypoints[6][1]) / 2
        hips_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knees_y = (keypoints[13][1] + keypoints[14][1]) / 2

        if abs(head_y - hips_y) > 0.5 * abs(hips_y - knees_y):
            return "Standing"
        elif abs(head_y - hips_y) > 0.2 * abs(hips_y - knees_y):
            return "Sitting"
        else:
            return "Lying Down"

    def classify_body_position(self, keypoints):
        """
        Classify body position based on keypoints.

        Args:
            keypoints (ndarray): Array of shape (num_keypoints, 3).

        Returns:
            str: One of "Standing", "Sitting", "Lying Down", or "Unknown".
        """
        # Define keypoint indices based on COCO format
        # COCO has 17 keypoints: 0 - nose, 1 - left eye, 2 - right eye, 3 - left ear,
        # 4 - right ear, 5 - left shoulder, 6 - right shoulder,
        # 7 - left elbow, 8 - right elbow, 9 - left wrist, 10 - right wrist,
        # 11 - left hip, 12 - right hip, 13 - left knee, 14 - right knee,
        # 15 - left ankle, 16 - right ankle
        # For simplicity, we'll use average positions for shoulders, hips, knees

        # Extract keypoints
        nose = keypoints[0][:2]
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]

        # Average positions
        shoulders = (left_shoulder + right_shoulder) / 2
        hips = (left_hip + right_hip) / 2
        knees = (left_knee + right_knee) / 2

        # Calculate vertical differences
        shoulder_hip_diff = abs(shoulders[1] - hips[1])
        hip_knee_diff = abs(hips[1] - knees[1])
        shoulder_nose_diff = abs(shoulders[1] - nose[1])

        # Calculate horizontal differences to assess alignment
        horizontal_diff = abs(shoulders[0] - hips[0]) + abs(hips[0] - knees[0])

        # Define thresholds (these may need to be adjusted based on actual data and image scale)
        vertical_threshold_standing = 100  # Example value
        vertical_threshold_sitting = 50    # Example value
        horizontal_threshold = 30          # Example value

        # Determine posture
        if shoulder_hip_diff > vertical_threshold_standing and hip_knee_diff > vertical_threshold_standing:
            return "Standing"
        elif shoulder_hip_diff > vertical_threshold_standing and hip_knee_diff < vertical_threshold_sitting:
            return "Sitting"
        elif horizontal_diff < horizontal_threshold:
            return "Lying Down"
        else:
            return "Unknown"

    def calculate_orientation(self, keypoints):
        """
        Calculate the orientation of the person in radians.

        Args:
            keypoints (ndarray): Array of shape (num_keypoints, 3 (x, y, confidence)).

        Returns:
            float: Orientation in radians.
        """
        # Use shoulders to calculate orientation
        point1 = keypoints[5][:2]  # Left shoulder  We only need the (x, y) coordinates for this calculation
        point2 = keypoints[6][:2]  # Right shoulder

        # Calculate the difference between the points
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]

        # Calculate the angle in radians
        angle_radians = np.arctan2(delta_y, delta_x)
        # An angle of 0 radians means the line is perfectly horizontal, with the right keypoint to the right of the left keypoint.
        # A positive angle indicates a counterclockwise rotation from the horizontal.
        # A negative angle indicates a clockwise rotation from the horizontal.
        return angle_radians

    def is_facing_towards_camera(self, keypoints):
        """
        Determine if the person is facing towards or away from the camera.

        Args:
            keypoints (ndarray): Array of shape (num_keypoints, 3).

        Returns:
            str: "Facing Towards" or "Facing Away".
        """
        # Extract keypoints
        nose = keypoints[0][:2]
        left_eye = keypoints[1][:2]
        right_eye = keypoints[2][:2]
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]

        # Check if keypoints are within the horizontal bounds of the shoulders
        shoulders_center = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

        if (left_shoulder[0] <= nose[0] <= right_shoulder[0]) and (left_shoulder[0] <= left_eye[0] <= right_shoulder[0]) and (left_shoulder[0] <= right_eye[0] <= right_shoulder[0]):
            return "Facing Towards"
        elif nose[0] < left_shoulder[0] or nose[0] > right_shoulder[0]:
            return "Facing Away"
        else:
            return "Facing Away"
