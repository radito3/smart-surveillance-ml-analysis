import numpy as np
import torch

from messaging.aggregate_consumer import AggregateConsumer
from messaging.broker_interface import Broker
from messaging.producer import Producer


class HumanObjectInteractionAnalyzer(Producer, AggregateConsumer):

    def __init__(self, broker: Broker):
        Producer.__init__(self, broker)
        AggregateConsumer.__init__(self, broker, ['object_detection_results', 'pose_detection_results'])
        self.interaction_durations: dict[int, int] = {}

    def get_name(self) -> str:
        return 'human-object-interaction-app'

    # might not be the most optimized implementation, but it doesn't contain any NN model, just in-memory calculations
    def consume_message(self, message: dict[str, any]):
        yolo_results = message['object_detection_results']
        pose_results = message['pose_detection_results']
        people = [(bbox, track_id) for bbox, track_id, cls, _ in yolo_results if cls == 0]
        objects = [(bbox, cls) for bbox, _, cls, _ in yolo_results if cls != 0]

        results: list[list[int]] = []
        for person_bbox, track_id in people:
            person_kpts = self.find_person_kpts(pose_results, person_bbox)
            track_id_key: int = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
            interacting_with_object: bool = False

            for object_bbox, cls in objects:
                if self.is_object_held(person_bbox, object_bbox, person_kpts[9:11]):
                    interacting_with_object = True
                    if track_id_key in self.interaction_durations:
                        self.interaction_durations[track_id_key] += 1
                    else:
                        self.interaction_durations[track_id_key] = 1
                    # is_interacting, interaction_duration, detected_object_class_index
                    results.append([1, self.interaction_durations[track_id_key], cls])
                    break  # only account for a single object being interacted with

            # ensure the HOI vector is the same size as the YOLO vector
            if not interacting_with_object:
                results.append([0, 0, -1])

        self.clear_people_no_longer_in_frame(people)
        self.produce_value('hoi_results', results)

    def cleanup(self):
        self.produce_value('hoi_results', None)

    def find_person_kpts(self, pose_results, person_bbox):
        for kpts in pose_results:
            if self.match_bbox_keypoints(person_bbox, kpts):
                return kpts

    @staticmethod
    def is_keypoint_in_bbox(keypoint, bbox) -> bool:
        x_min, y_min, x_max, y_max = bbox
        x_kp, y_kp = keypoint

        return x_min <= x_kp <= x_max and y_min <= y_kp <= y_max

    def match_bbox_keypoints(self, bbox, keypoints, threshold=0.5) -> bool:
        keypoints_in_bbox = sum(self.is_keypoint_in_bbox(kp, bbox) for kp in keypoints)
        total_keypoints = len(keypoints)

        proportion_in_bbox = keypoints_in_bbox / total_keypoints
        return proportion_in_bbox >= threshold

    def is_object_held(self, person_bbox, object_bbox, hand_keypoints) -> bool:
        iou = self.calculate_iou(person_bbox, object_bbox)

        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        object_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])
        dynamic_iou_threshold = 0.1 * (object_area / person_area)  # person-to-object size ratio

        hand_near_object = self.calculate_hand_proximity(hand_keypoints, object_bbox, person_bbox)

        return iou > dynamic_iou_threshold and hand_near_object

    def clear_people_no_longer_in_frame(self, people):
        for track_id in list(self.interaction_durations.keys()):
            present: bool = False
            for _, _track_id in people:
                track_id_key = int(_track_id.item()) if isinstance(_track_id, torch.Tensor) else int(_track_id)
                if track_id_key == track_id:
                    present = True
                    break
            if not present:
                del self.interaction_durations[track_id]

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

        area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

        return intersection_area / (area1 + area2 - intersection_area)

    @staticmethod
    def calculate_hand_proximity(hand_keypoints, object_bbox, person_bbox):
        # hand_keypoints format: [(wrist_x, wrist_y)]
        wrist_left, wrist_right = hand_keypoints

        obj_center_x = (object_bbox[0] + object_bbox[2]) / 2
        obj_center_y = (object_bbox[1] + object_bbox[3]) / 2

        # Calculate Euclidean distances from wrist keypoints to the object center
        dist_left = np.sqrt((wrist_left[0] - obj_center_x) ** 2 + (wrist_left[1] - obj_center_y) ** 2)
        dist_right = np.sqrt((wrist_right[0] - obj_center_x) ** 2 + (wrist_right[1] - obj_center_y) ** 2)

        # Normalize distances based on the person bounding box height (or width)
        person_height = person_bbox[3] - person_bbox[1]

        normalized_dist_left = dist_left / person_height
        normalized_dist_right = dist_right / person_height

        threshold = 0.1
        return normalized_dist_left < threshold or normalized_dist_right < threshold
