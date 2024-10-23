import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGPooling
from torch_geometric.data import Data
from itertools import zip_longest
from math import sqrt

from messaging.aggregate_consumer import AggregateConsumer
from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer
from util.device import get_device


class GraphNetWithSAGPooling(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, pooling_channels=16, pooling_ratio=0.8, global_pool_type='mean'):
        super(GraphNetWithSAGPooling, self).__init__()
        self.conv1 = GCNConv(node_features, pooling_channels)
        # Self-Attention Graph Pooling
        self.filter_pool = SAGPooling(pooling_channels, ratio=pooling_ratio)
        self.conv2 = GCNConv(pooling_channels, hidden_dim)
        # Ensures a fixed-size output
        self.global_pool = global_mean_pool if global_pool_type == 'mean' else global_max_pool

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.filter_pool(x, edge_index, edge_weight, batch)
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        x = self.global_pool(x, batch)  # Enforce a fixed-sized output
        return x


class GraphBasedLSTMClassifier(torch.nn.Module):
    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 16,
                 pooling_channels: int = 16,
                 pooling_ratio: float = 0.8,
                 global_pool_type: str = 'mean',
                 lstm_layers: int = 1):
        torch.nn.Module.__init__(self)
        self.gnn = GraphNetWithSAGPooling(node_features, hidden_dim, pooling_channels, pooling_ratio, global_pool_type)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
                            dropout=0.2 if lstm_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.width, self.height = 640, 640
        self.prev_detections = []
        self.velocities = {}

    def set_dimensions(self, dims: tuple[float, float]):
        self.width, self.height = dims

    def forward(self, graph_data_sequences):
        # graph_data_sequences is a list of graph data for each time step
        embeddings = []
        for data in graph_data_sequences:
            data.to(get_device())
            graph_embedding = self.gnn(data)
            embeddings.append(graph_embedding.unsqueeze(1))  # Add sequence dimension
        embeddings = torch.cat(embeddings, dim=1)  # Shape: (batch_size, sequence_length, features)

        lstm_out, _ = self.lstm(embeddings)
        predictions = self.output_layer(lstm_out[:, -1, :])  # Classify based on the output of the last time step
        return torch.sigmoid(predictions)

    def create_graph(self, pose_results, hoi_results, detected_activities):
        nodes, edges, edge_weights = self._extract_graph(pose_results, hoi_results, detected_activities)
        return Data(x=torch.tensor(nodes, dtype=torch.float),
                    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(edge_weights, dtype=torch.float)
                    )

    def _extract_graph(self, pose_results, hoi_results, detected_activities):
        """
        The node features are:
         - is_interacting,                   # Binary: Is there interaction or not
         - interaction_duration,             # How long the interaction has lasted
         - detected_object_class_index       # YOLO object class index (e.g., 0 for person, 39 for bottle, etc.)
         - One-hot encoded body positions
            standing, sitting, laying_down, crouching, bending_over, walking, ...
         - body_orientation,                 # Where a person is facing
         - velocity,                         # Current velocity (based on keypoints)
         - average_velocity,                 # Average velocity over a sequence of time steps
         - kinetics_400_activity             # Activity from Kinetics 400 dataset (encoded as an index)

        The edges have a higher index if people are close together multiplied by if they are facing each other.
        """

        detections = [(box_tensor, box_id) for box_tensor, box_id, _ in pose_results]

        centroids_current = np.array([self._calc_centroid(bbox) for bbox, _ in detections])

        # Velocity can be the magnitude of change in position
        velocities = self._calc_velocities(detections)

        self._save_velocities(zip(detections, velocities))

        average_velocities = self._calc_avg_velocities(detections)

        # Direction can be encapsulated as the angle of orientation
        orientations = np.array([self._calculate_orientation(keypoints) for _, _, keypoints in pose_results], dtype=np.float32)

        body_positions = np.array([self._classify_position(keypoints) for _, _, keypoints in pose_results], dtype=np.float32)

        activities = np.array(detected_activities, dtype=np.float32).flatten()

        # FIXME: the paddings are still insufficient sometimes and np.hstack throws an error
        if len(activities) < len(detections):
            for _ in range(len(detections) - len(activities)):
                activities = np.append(activities, -1)
        if len(activities) > len(detections):
            activities = np.delete(activities, len(activities) - 1)

        features = np.hstack((body_positions.reshape(-1, 1), velocities.reshape(-1, 1),
                              average_velocities.reshape(-1, 1), orientations.reshape(-1, 1),
                              activities.reshape(-1, 1)))

        edges = []
        edge_weights = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                """
                edge weight = 1/(distance / sqrt(width ^ 2 + height ^ 2)) * (2 - cos(|theta1 - theta2| mod 2 * pi))
                """
                euclidean_distance = np.linalg.norm(centroids_current[i] - centroids_current[j]).__float__()
                norm_distance = self.__normalize_distance(euclidean_distance)
                phi = np.abs(orientations[i] - orientations[j]) % (2 * np.pi)
                f_phi = 2 - np.cos(phi)
                edge_weight = 1 / (norm_distance * f_phi)

                # make a complete (fully-connected) graph
                edges.append((i, j))
                edges.append((j, i))  # because the graph is undirected
                edge_weights.append(edge_weight)

        self.prev_detections = detections
        return features, edges, edge_weights

    def __normalize_distance(self, distance: float) -> float:
        return distance / sqrt(self.width ** 2 + self.height ** 2)

    @staticmethod
    def __contains_by_id(detections, track_id) -> bool:
        for _, _id in detections:
            id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
            id_cmp_val = int(_id.item()) if isinstance(_id, torch.Tensor) else int(_id)
            if id_cmp_val == id_val:
                return True
        return False

    def _calc_velocities(self, detections) -> np.array:
        if len(self.prev_detections) > 0:
            if len(detections) > len(self.prev_detections):
                generator = zip_longest(detections, self.prev_detections,
                                        fillvalue=(np.zeros(4, dtype=np.float32), -1))
            else:
                generator = zip(detections,
                                filter(lambda tpl: self.__contains_by_id(detections, tpl[1]), self.prev_detections))
            displacement = [self._calc_centroid(current_bbox) - self._calc_centroid(prev_bbox)
                            if current_id == prev_id else np.zeros(2, dtype=np.float32)
                            for (current_bbox, current_id), (prev_bbox, prev_id) in generator
                            ]

            return np.linalg.norm(displacement, axis=1)
        else:
            return np.zeros(len(detections))

    def _save_velocities(self, bbox_velocities_tuples):
        for (_, track_id), velocity in bbox_velocities_tuples:
            id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
            if id_val in self.velocities:
                self.velocities[id_val].append(velocity)
            else:
                self.velocities.update({id_val: [velocity]})

    def _calc_avg_velocities(self, detections) -> np.array:
        def __contains_by_id(track_id: int) -> bool:
            for _, _id in detections:
                if _id == track_id:
                    return True
            return False

        return np.array(
            [np.average(velocities) for track_id, velocities in self.velocities.items() if __contains_by_id(track_id)],
            dtype=np.float32)

    @staticmethod
    def _calc_centroid(bbox) -> np.ndarray:
        # bounding box in xyxy format [xmin, ymin, xmax, ymax]
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

    @staticmethod
    def _classify_position(keypoints) -> int:
        head_y = keypoints[0][1]
        # shoulders_y = (keypoints[5][1] + keypoints[6][1]) / 2
        hips_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knees_y = (keypoints[13][1] + keypoints[14][1]) / 2

        # TODO: adapt this
        # head = np.array(keypoints['head'])
        # left_shoulder = np.array(keypoints['left_shoulder'])
        # right_shoulder = np.array(keypoints['right_shoulder'])
        # left_hip = np.array(keypoints['left_hip'])
        # right_hip = np.array(keypoints['right_hip'])
        # left_knee = np.array(keypoints['left_knee'])
        # right_knee = np.array(keypoints['right_knee'])
        # left_ankle = np.array(keypoints['left_ankle'])
        # right_ankle = np.array(keypoints['right_ankle'])
        #
        # # Average keypoints for symmetry
        # shoulders = (left_shoulder + right_shoulder) / 2
        # hips = (left_hip + right_hip) / 2
        # knees = (left_knee + right_knee) / 2
        # ankles = (left_ankle + right_ankle) / 2
        #
        # # Calculate vertical distances between keypoints
        # shoulder_hip_dist = np.abs(shoulders[1] - hips[1])
        # hip_knee_dist = np.abs(hips[1] - knees[1])
        # knee_ankle_dist = np.abs(knees[1] - ankles[1])
        # head_ankle_dist = np.abs(head[1] - ankles[1])
        #
        # # Thresholds (can be adjusted based on the use case)
        # standing_threshold = 0.4 * head_ankle_dist  # Standing requires vertical alignment
        # sitting_threshold = 0.2 * head_ankle_dist  # Sitting makes hips close to knees
        # laying_down_threshold = 50  # Low vertical distance between all points
        #
        # # Initialize one-hot encoding for 5 body positions: [standing, sitting, laying down, crouching, bending over]
        # one_hot_position = [0, 0, 0, 0, 0]
        #
        # # Detect body position
        # if knee_ankle_dist > standing_threshold and hip_knee_dist > standing_threshold:
        #     # Standing: knees are far from ankles and hips are far from knees
        #     one_hot_position[0] = 1  # Standing
        #
        # elif hip_knee_dist < sitting_threshold and knee_ankle_dist > sitting_threshold:
        #     # Sitting: hips are closer to knees but knees are far from ankles
        #     one_hot_position[1] = 1  # Sitting
        #
        # elif shoulder_hip_dist < laying_down_threshold and hip_knee_dist < laying_down_threshold:
        #     # Laying down: All major points are close vertically
        #     one_hot_position[2] = 1  # Laying down
        #
        # elif hip_knee_dist < sitting_threshold and knee_ankle_dist < standing_threshold:
        #     # Crouching: Hips are close to knees and knees close to ankles
        #     one_hot_position[3] = 1  # Crouching
        #
        # elif shoulders[1] > hips[1] and hip_knee_dist > standing_threshold:
        #     # Bending over: Shoulders are significantly lower than hips
        #     one_hot_position[4] = 1  # Bending over
        #
        # return one_hot_position

        if abs(head_y - hips_y) > 0.5 * abs(hips_y - knees_y):
            return 0  # Standing
        elif abs(head_y - hips_y) > 0.2 * abs(hips_y - knees_y):
            return 1  # Sitting
        else:
            return -1  # Lying Down

    @staticmethod
    def _calculate_orientation(keypoints):
        """
        Calculate the orientation of the person in radians.

        Args:
            keypoints (ndarray): Array of shape (num_keypoints, 3 (x, y, confidence)).

        Returns:
            float: Orientation in radians.
        """
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]

        delta_y = right_shoulder[1] - left_shoulder[1]
        delta_x = right_shoulder[0] - left_shoulder[0]

        # if angle_radians < 0:
        #     angle_radians += 2*np.pi  # phase shift, so that values are only positive

        # An angle of 0 radians means the line is perfectly horizontal
        # A positive angle indicates a counterclockwise rotation from the horizontal
        # A negative angle indicates a clockwise rotation from the horizontal
        return np.arctan2(delta_y, delta_x)


class CompositeBehaviouralClassifier(Producer, AggregateConsumer):
    def __init__(self,
                 broker: Broker,
                 node_features: int,
                 window_size: int,
                 window_step: int,
                 hidden_dim: int = 16,
                 pooling_channels: int = 16,
                 pooling_ratio: float = 0.8,
                 global_pool_type: str = 'mean',
                 lstm_layers: int = 1):
        Producer.__init__(self, broker)
        AggregateConsumer.__init__(self, broker,
                                   ['pose_detection_results', 'hoi_results', 'activity_detection_results'],
                                   pose_detection_results=window_size, hoi_results=window_size, step=window_step)
        self.classifier = None
        self.is_injected: bool = False

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.pooling_channels = pooling_channels
        self.pooling_ratio = pooling_ratio
        self.global_pool_type = global_pool_type
        self.lstm_layers = lstm_layers

    def inject_model(self, model: GraphBasedLSTMClassifier):
        self.classifier = model
        self.is_injected = True

    def init(self):
        if not self.is_injected:
            self.classifier = GraphBasedLSTMClassifier(self.node_features, self.hidden_dim, self.pooling_channels,
                                                       self.pooling_ratio, self.global_pool_type, self.lstm_layers).to(
                get_device())
            if 'GRAPH_LSTM_WEIGHTS_PATH' in os.environ:
                pretrained_weights_path = os.environ['GRAPH_LSTM_WEIGHTS_PATH']
                if len(pretrained_weights_path) != 0:
                    self.classifier.load_state_dict(torch.load(pretrained_weights_path, map_location=get_device()))
            # only on CUDA due to: https://github.com/pytorch/pytorch/issues/125254
            self.classifier.compile() if torch.cuda.is_available() else None
        temp_consumer = OneShotConsumer(self.broker, 'video_dimensions', self)
        temp_consumer.run()

    def get_name(self) -> str:
        return 'graph-lstm-classifier-app'

    def consume_message(self, message: dict[str, any]):
        pose_buffer = message['pose_detection_results']
        hoi_buffer = message['hoi_results']
        detected_activities = message['activity_detection_results']

        graph_data_sequences = [
            self.classifier.create_graph(pose_results, hoi_results, detected_activities) for
            pose_results, hoi_results in zip(pose_buffer, hoi_buffer)
        ]

        if not self.classifier.training:
            with torch.no_grad():
                predictions = self.classifier(graph_data_sequences)
        else:
            predictions = self.classifier(graph_data_sequences)

        result = torch.flatten(predictions).cpu().item()
        self.classifier.velocities.clear()
        self.produce_value('classification_results', result)

    def cleanup(self):
        self.produce_value('classification_results', None)


class OneShotConsumer(Consumer):

    def __init__(self, broker: Broker, topic: str, stream_app: CompositeBehaviouralClassifier):
        super().__init__(broker, topic)
        self.stream_app = stream_app

    def get_name(self) -> str:
        return 'one-shot-consumer'

    def consume_message(self, message: tuple[float, float]):
        self.stream_app.classifier.set_dimensions(message)
