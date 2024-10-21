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
        # FIXME: ensure uniqueness of pairings - currently more than one person is matched to a pose
        people_to_poses = self.match_bboxes_to_keypoints(people, pose_results)

        results: list[list[int]] = []
        for i, (person_bbox, track_id) in enumerate(people):
            # print(f"bbox idx: {i} pose idx: {people_to_poses[i]}")
            person_kpts = pose_results[people_to_poses[i]]
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

    @staticmethod
    def bbox_centroid(bbox) -> tuple[float, float]:
        x_min, y_min, x_max, y_max = bbox
        x_centroid = (x_min + x_max) / 2
        y_centroid = (y_min + y_max) / 2
        return x_centroid, y_centroid

    @staticmethod
    def keypoints_centroid(keypoints) -> tuple[float, float]:
        x_coords = [kp[0] for kp in keypoints]
        y_coords = [kp[1] for kp in keypoints]

        # Calculate the centroid
        x_centroid = np.mean(x_coords).__float__()
        y_centroid = np.mean(y_coords).__float__()

        return x_centroid, y_centroid

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # apply Geometric Centroid Matching to find which pose maps to which person
    def match_bboxes_to_keypoints(self, bboxes, poses):
        bbox_centroids = [self.bbox_centroid(bbox) for bbox, _ in bboxes]
        pose_centroids = [self.keypoints_centroid(pose) for pose in poses]

        matches: dict[int, int] = {}

        # For each bounding box, find the closest keypoints centroid
        for i, bbox_c in enumerate(bbox_centroids):
            min_distance = float('inf')
            best_pose_idx = -1

            for j, pose_c in enumerate(pose_centroids):
                dist = self.euclidean_distance(bbox_c, pose_c)
                if dist < min_distance:
                    min_distance = dist
                    best_pose_idx = j

            matches[i] = best_pose_idx

        return matches
