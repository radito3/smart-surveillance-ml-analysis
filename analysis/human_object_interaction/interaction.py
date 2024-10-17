import cv2
import numpy as np


class HumanObjectInteractionAnalyzer:

    def analyze(self, frame: cv2.typing.MatLike, *args, **kwargs) -> list[int]:
        # integrate with YOLO-pose results
        return []

    # Helper function to calculate the Euclidean distance between two points
    @staticmethod
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # IoU Calculation Function
    @staticmethod
    def calculate_iou(bbox1, bbox2):
        # bbox format: [x_min, y_min, x_max, y_max]
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2

        # Calculate intersection
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)

        intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

        # Calculate areas
        area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

        # Calculate IoU
        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou

    # Pose Proximity Check (hand distance from object)
    @staticmethod
    def calculate_hand_proximity(hand_keypoints, object_bbox, person_bbox):
        # hand_keypoints format: [(wrist_x, wrist_y)]
        # object_bbox format: [x_min, y_min, x_max, y_max]
        wrist_left, wrist_right = hand_keypoints

        # Object bounding box center
        obj_center_x = (object_bbox[0] + object_bbox[2]) / 2
        obj_center_y = (object_bbox[1] + object_bbox[3]) / 2

        # Calculate Euclidean distances from wrist keypoints to the object center
        dist_left = np.sqrt((wrist_left[0] - obj_center_x) ** 2 + (wrist_left[1] - obj_center_y) ** 2)
        dist_right = np.sqrt((wrist_right[0] - obj_center_x) ** 2 + (wrist_right[1] - obj_center_y) ** 2)

        # Normalize distances based on the person bounding box height (or width)
        person_height = person_bbox[3] - person_bbox[1]

        normalized_dist_left = dist_left / person_height
        normalized_dist_right = dist_right / person_height

        # If either hand is close enough (below some threshold, e.g., 0.1), return True
        threshold = 0.1  # Distance threshold as a fraction of person height
        return normalized_dist_left < threshold or normalized_dist_right < threshold

    # Main function to check if object is held
    def is_object_held(self, person_bbox, object_bbox, hand_keypoints):
        # Step 1: Calculate IoU between person and object
        iou = self.calculate_iou(person_bbox, object_bbox)

        # Step 2: Set dynamic IoU threshold based on person-to-object size ratio
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        object_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])
        dynamic_iou_threshold = 0.1 * (object_area / person_area)

        # Step 3: Calculate hand proximity to the object
        hand_near_object = self.calculate_hand_proximity(hand_keypoints, object_bbox, person_bbox)

        # Step 4: Combine IoU and hand proximity to decide if object is held
        return iou > dynamic_iou_threshold and hand_near_object

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
