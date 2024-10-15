import numpy as np
import torch
import torch.nn as nn
from torch.utils.hipify.hipify_python import value
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGPooling
from torch_geometric.data import Data
from itertools import zip_longest
from math import sqrt

from analysis.types import AnalysisType
from classification.classifier import Classifier
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.filter_pool(x, edge_index, None, batch)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.global_pool(x, batch)  # Enforce a fixed-sized output
        return x


class GraphBasedLSTMClassifier(torch.nn.Module, Classifier):
    def __init__(self,
                 node_features: int,
                 window_size: int,
                 window_step: int,
                 dimensions: tuple[float, float] = (640, 640),
                 hidden_dim=16,
                 pooling_channels=16,
                 pooling_ratio=0.8,
                 global_pool_type='mean',
                 lstm_layers=1):
        torch.nn.Module.__init__(self)

        self.gnn = GraphNetWithSAGPooling(node_features, hidden_dim, pooling_channels, pooling_ratio, global_pool_type)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
                            dropout=0.2 if lstm_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Outputting a single probability

        self.window_size = window_size
        self.window_step = window_step  # every N frames move to a new window
        self.width, self.height = dimensions
        self.prev_detections: list[torch.Tensor] = []
        self.yolo_buffer = []
        self.pose_buffer = []
        self.velocities = {}
        self.activities = None

    def set_dimensions(self, dims: tuple[float, float]):
        self.width, self.height = dims

    def set_window_size(self, size: int):
        self.window_size = size

    def set_window_step(self, step: int):
        self.window_step = step

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

    def classify_as_suspicious(self, dtype: AnalysisType, vector: list[any]) -> float:
        if (len(self.yolo_buffer) < self.window_size
                or len(self.pose_buffer) < self.window_size
                or self.activities is None):
            self.yolo_buffer.append(vector) if dtype == AnalysisType.PersonDetection else None
            self.pose_buffer.append(vector) if dtype == AnalysisType.PoseEstimation else None
            self.activities = vector if dtype == AnalysisType.ActivityDetection else None
            return 0

        graph_data_sequences = [self._create_graph(yolo_results, pose_results) for yolo_results, pose_results in
                                zip(self.yolo_buffer, self.pose_buffer)]

        self.yolo_buffer = self.yolo_buffer[self.window_step:]
        self.pose_buffer = self.pose_buffer[self.window_step:]
        self.activities = None

        predictions: torch.Tensor = self.forward(graph_data_sequences)
        return torch.flatten(predictions.cpu()).item()

    def _create_graph(self, yolo_results, pose_results):
        nodes, edges, edge_weights = self._extract_graph(yolo_results, pose_results)
        return Data(x=torch.tensor(nodes, dtype=torch.float),
                    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(edge_weights, dtype=torch.float)
                    )

    def _extract_graph(self, yolo_results, pose_results):
        """
        The node features are:
         - orientation (where a person is facing)
         - velocity
         - average velocity
         - body position - standing, sitting, laying
         - activity index from Kinetics-400

        The edges have a higher index if people are close together multiplied by if they are facing each other.
        """

        # we don't need the confidence scores here
        detections = [(box_tensor, box_id) for box_tensor, box_id, _ in yolo_results]

        centroids_current = np.array([self._calc_centroid(bbox) for bbox, _ in detections])

        # Velocity can be the magnitude of change in position
        velocities = self._calc_velocities(detections)

        self._save_velocities(zip(detections, velocities))

        average_velocities = self._calc_avg_velocities(detections)

        # Direction can be encapsulated as the angle of orientation
        orientations = np.array([self._calculate_orientation(keypoints) for keypoints in pose_results], dtype=np.float32)

        body_positions = np.array([self._classify_position(keypoints) for keypoints in pose_results], dtype=np.float32)

        activities = np.array(self.activities, dtype=np.float32).flatten()

        # these paddings aren't great...
        if len(yolo_results) > len(pose_results):
            for _ in range(len(yolo_results) - len(pose_results)):
                orientations = np.append(orientations, 0)
                body_positions = np.append(body_positions, 0)

        if len(pose_results) > len(yolo_results):
            for _ in range(len(pose_results) - len(yolo_results)):
                velocities = np.append(velocities, 0)
                average_velocities = np.append(average_velocities, 0)
                centroids_current = np.append(centroids_current, np.array([0, 0]))

        if len(self.activities) < max(len(yolo_results), len(pose_results)):
            for _ in range(max(len(yolo_results), len(pose_results)) - len(self.activities)):
                activities = np.append(activities, -1)
        if len(self.activities) > max(len(yolo_results), len(pose_results)):
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
