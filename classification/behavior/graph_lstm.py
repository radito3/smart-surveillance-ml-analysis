import os
from threading import Event

import numpy as np
import torch
from torch.nn import Linear, LSTM
from torch_geometric.nn import GATConv, SAGPooling, AttentionalAggregation, MaxAggregation
from torch_geometric.data import Data

from messaging.processor import MessageProcessor
from util.device import get_device

class GraphNetWithAttentionPooling(torch.nn.Module):
    def __init__(self, node_features, hidden_dim,
                 pooling_channels=16, pooling_ratio=0.8, attention_heads=4, node_threshold=30):
        super(GraphNetWithAttentionPooling, self).__init__()
        self.node_threshold = node_threshold  # Dynamic pooling threshold
        self.conv1 = GATConv(node_features, pooling_channels // attention_heads, heads=attention_heads, edge_dim=1,
                             dropout=0.1, add_self_loops=False)
        self.filter_pool = SAGPooling(pooling_channels, ratio=pooling_ratio)  # Self-Attention Graph Pooling
        self.conv2 = GATConv(pooling_channels, hidden_dim, heads=1, edge_dim=1, dropout=0.1, add_self_loops=False)
        self.attention_pool = AttentionalAggregation(Linear(hidden_dim, 1))
        self.max_pool = MaxAggregation()  # for catching outliers

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        if x.shape[0] > self.node_threshold:
            x, edge_index, edge_attr, _, _, _ = self.filter_pool(x, edge_index, edge_attr)
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        index = edge_index.new_zeros(x.size(0))  # shape: (num_nodes, hidden_dim)
        attention_scored = self.attention_pool(x, index)
        max_x = self.max_pool(x, index)
        # Combine the attention-scored and max values for the graph to give the LSTM more context
        return torch.cat((attention_scored, max_x), dim=1)


class GraphBasedLSTMClassifier(torch.nn.Module):
    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 16,
                 pooling_channels: int = 16,
                 pooling_ratio: float = 0.8,
                 attention_heads: int = 4,
                 node_threshold: int = 30,
                 lstm_layers: int = 1):
        torch.nn.Module.__init__(self)
        self.gnn = GraphNetWithAttentionPooling(node_features, hidden_dim, pooling_channels, pooling_ratio,
                                                attention_heads, node_threshold)
        # double the input size because we are concatenating the attention and max pooling results
        self.lstm = LSTM(hidden_dim * 2, hidden_dim, num_layers=lstm_layers, batch_first=True,
                         bidirectional=True, dropout=0.2 if lstm_layers > 1 else 0)
        self.output_layer = Linear(hidden_dim * 2, 1)  # double because of the BiLSTM

        self.width, self.height = 640, 640
        self.alpha = 2.0  # tunable edge weight parameter
        self.prev_detections = []
        self.velocities = {}

    def set_dimensions(self, dims: tuple[float, float]):
        self.width, self.height = dims

    def forward(self, graph_data_sequences):
        # graph_data_sequences is a list of graph data for each time step
        embeddings = []
        for data in graph_data_sequences:
            data.to(get_device())
            graph_embedding = self.gnn(data) if len(data.x) >= 2 else torch.empty([1, 17], device=get_device())
            embeddings.append(graph_embedding.unsqueeze(1))  # Add sequence dimension
        embeddings = torch.cat(embeddings, dim=1)  # Shape: (batch_size, sequence_length, features)

        lstm_out, _ = self.lstm(embeddings)
        predictions = self.output_layer(lstm_out[:, -1, :])  # Classify based on the output of the last time step
        return torch.sigmoid(predictions)

    def create_graph(self, pose_results, hoi_results, detected_activities):
        if len(pose_results) == 0:
            return Data(x=torch.empty((0,)))
        nodes, edges, edge_weights = self._extract_graph(pose_results, hoi_results, detected_activities)
        return Data(x=torch.tensor(nodes, dtype=torch.float),
                    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(edge_weights, dtype=torch.float)
                    )

    # TODO: try to move all operations to the GPU -> convert any NumPy methods and arrays to PyTorch equivalents
    def _extract_graph(self, pose_results, hoi_results, detected_activities):
        """
        The node features are:
         - is_interacting,                   # Binary: Is there interaction or not
         - interaction_duration,             # How long the interaction has lasted
         - detected_object_class_index       # YOLO object class index (e.g., 0 for person, 39 for bottle, etc.)
         - One-hot encoded body positions:
            [standing, sitting, laying down, crouching, bending over]
         - body_orientation                  # Where a person is facing (as [sin(phi), cos(phi)])
         - velocity,                         # Current velocity (in bbox centroid displacement between frames)
         - average_velocity,                 # Average velocity over the sequence of time steps
         - kinetics_400_activity             # Activity from Kinetics 400 dataset (encoded as an index)

        The edges have a higher index if people are close together multiplied by if they are facing each other.
        """
        # Velocity can be the magnitude of change in position
        velocities = self._calc_velocities(pose_results)
        # TODO: scale velocities so they are within [0, 1]
        self._save_velocities(zip(pose_results, velocities))

        average_velocities = self._calc_avg_velocities(pose_results)
        # TODO: scale avg velocities so they are within [0, 1]
        # Direction can be encapsulated as the angle of orientation
        orientations = np.array([self._calculate_orientation(keypoints) for _, _, keypoints in pose_results],
                                dtype=np.float32)
        sin_orients = np.sin(orientations)
        cos_orients = np.cos(orientations)
        orientations_encoded = np.stack([sin_orients, cos_orients], axis=-1)

        body_positions = np.array([self._classify_position(keypoints) for _, _, keypoints in pose_results],
                                  dtype=np.float32)

        activities = np.array([activity_idx for track_id, activity_idx in detected_activities if
                               self.__contains_by_id(pose_results, track_id)], dtype=np.float32)
        if len(activities) < len(pose_results):
            activities = np.append(activities, [0 for _ in range(len(pose_results) - len(activities))])

        object_interactions = np.array([results for track_id, results in hoi_results if
                                        self.__contains_by_id(pose_results, track_id)], dtype=np.float32)
        if len(object_interactions) < len(pose_results):
            object_interactions = np.vstack([object_interactions]
                                            + [[0, 0, -1] for _ in range(len(pose_results) - len(object_interactions))])
        # TODO: apply scaling to the interation duration so it's within [0, 1]

        features = np.hstack((object_interactions.reshape(object_interactions.shape[0], -1),
                              body_positions.reshape(body_positions.shape[0], -1),
                              orientations_encoded.reshape(orientations_encoded.shape[0], -1),
                              velocities.reshape(-1, 1),
                              average_velocities.reshape(-1, 1),
                              activities.reshape(-1, 1)))

        centroids_current = np.array([self._calc_centroid(bbox) for bbox, _, _ in pose_results])
        edges = []
        edge_weights = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                # Distance term: e^(-alpha * distance / scale)
                euclidean_distance = np.linalg.norm(centroids_current[i] - centroids_current[j]).__float__()
                scale = torch.sqrt(torch.tensor(self.width ** 2 + self.height ** 2, dtype=torch.float32))
                normalized_distance = euclidean_distance / scale
                distance_term = torch.exp(-self.alpha * normalized_distance)

                # Orientation term: 1 + cos(|theta1 - theta2| - pi)
                angle_diff = np.abs(orientations[i] - orientations[j])
                orientation_term = 1 + torch.cos(angle_diff - torch.pi)

                edge_weight = distance_term * orientation_term

                # make a symmetric (fully-connected) graph
                edges.append((i, j))
                edges.append((j, i))  # because the graph is undirected
                edge_weights.append(edge_weight)
                edge_weights.append(edge_weight)

        self.prev_detections = pose_results
        return features, edges, edge_weights

    @staticmethod
    def __contains_by_id(detections, track_id) -> bool:
        for _, _id, _ in detections:
            id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
            id_cmp_val = int(_id.item()) if isinstance(_id, torch.Tensor) else int(_id)
            if id_cmp_val == id_val:
                return True
        return False

    def _calc_velocities(self, detections) -> np.ndarray:
        if len(self.prev_detections) > 0:
            displacement: list[np.ndarray] = [np.zeros(2, dtype=np.float32)] * len(detections)
            for i, (bbox, track_id, _) in enumerate(detections):
                for prev_bbox, prev_tid, _ in self.prev_detections:
                    id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
                    prev_id_val = int(prev_tid.item()) if isinstance(prev_tid, torch.Tensor) else int(prev_tid)
                    if id_val == prev_id_val:
                        displacement[i] = self._calc_centroid(bbox) - self._calc_centroid(prev_bbox)
                        break

            return np.linalg.norm(displacement, axis=1)
        else:
            return np.zeros(len(detections))

    def _save_velocities(self, bbox_velocities_tuples):
        for (_, track_id, _), velocity in bbox_velocities_tuples:
            id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
            if id_val in self.velocities:
                self.velocities[id_val].append(velocity)
            else:
                self.velocities[id_val] = [velocity]

    def _calc_avg_velocities(self, detections) -> np.array:
        result: list[float] = [0.0] * len(detections)
        for i, (bbox, track_id, _) in enumerate(detections):
            for sum_track_id, velocities in self.velocities.items():
                id_val = int(track_id.item()) if isinstance(track_id, torch.Tensor) else int(track_id)
                if id_val == sum_track_id:
                    result[i] = np.average(velocities).__float__()
                    break
        return np.array(result, dtype=np.float32)

    @staticmethod
    def _calc_centroid(bbox) -> np.ndarray:
        # bounding box in xyxy format [xmin, ymin, xmax, ymax]
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

    @staticmethod
    def _classify_position(keypoints) -> list[int]:
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee
        # 15: Left Ankle 16: Right Ankle

        # Get relevant y-coordinates from the keypoints
        head_y = keypoints[0][1]
        shoulders_y = (keypoints[5][1] + keypoints[6][1]) / 2
        hips_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knees_y = (keypoints[13][1] + keypoints[14][1]) / 2
        ankles_y = (keypoints[15][1] + keypoints[16][1]) / 2

        # Calculate the total height from head to ankles
        head_ankle_dist = np.abs(head_y - ankles_y)

        # Compute relative vertical distances between keypoints
        shoulder_hip_dist = np.abs(shoulders_y - hips_y)
        hip_knee_dist = np.abs(hips_y - knees_y)
        knee_ankle_dist = np.abs(knees_y - ankles_y)

        # Proportional thresholds based on total body height (head to ankle)
        standing_threshold = 0.4 * head_ankle_dist  # Standing requires vertical alignment
        sitting_threshold = 0.2 * head_ankle_dist  # Sitting brings hips closer to knees
        laying_down_threshold = 0.1 * head_ankle_dist  # Low vertical distance between keypoints

        # Initialize one-hot encoding for 5 body positions: [standing, sitting, laying down, crouching, bending over]
        one_hot_position = [0, 0, 0, 0, 0]

        # Detect body position based on relative distances
        if knee_ankle_dist > standing_threshold and hip_knee_dist > standing_threshold:
            # Standing: knees are far from ankles, and hips are far from knees
            one_hot_position[0] = 1  # Standing

        elif hip_knee_dist < sitting_threshold < knee_ankle_dist:
            # Sitting: hips are closer to knees but knees are far from ankles
            one_hot_position[1] = 1  # Sitting

        elif shoulder_hip_dist < laying_down_threshold and hip_knee_dist < laying_down_threshold:
            # Laying down: All major points are close vertically
            one_hot_position[2] = 1  # Laying down

        elif hip_knee_dist < sitting_threshold and knee_ankle_dist < standing_threshold:
            # Crouching: hips are close to knees and knees close to ankles
            one_hot_position[3] = 1  # Crouching

        elif shoulders_y > hips_y and hip_knee_dist > standing_threshold:
            # Bending over: shoulders are significantly lower than hips
            one_hot_position[4] = 1  # Bending over

        return one_hot_position

    @staticmethod
    def _calculate_orientation(keypoints):
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


class CompositeBehaviouralClassifier(MessageProcessor):
    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 16,
                 pooling_channels: int = 16,
                 pooling_ratio: float = 0.8,
                 attention_heads: int = 4,
                 node_threshold: int = 30,
                 lstm_layers: int = 1):
        super().__init__()
        self.classifier = None
        self.is_injected: bool = False
        self.is_initialized: Event = Event()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.pooling_channels = pooling_channels
        self.pooling_ratio = pooling_ratio
        self.attention_heads = attention_heads
        self.node_threshold = node_threshold
        self.lstm_layers = lstm_layers

    def inject_model(self, model: GraphBasedLSTMClassifier):
        self.classifier = model
        self.is_injected = True

    def init(self):
        if not self.is_injected:
            self.classifier = GraphBasedLSTMClassifier(self.node_features, self.hidden_dim, self.pooling_channels,
                                                       self.pooling_ratio, self.attention_heads, self.node_threshold,
                                                       self.lstm_layers).to(get_device())
            if 'GRAPH_LSTM_WEIGHTS_PATH' in os.environ:
                pretrained_weights_path = os.environ['GRAPH_LSTM_WEIGHTS_PATH']
                if len(pretrained_weights_path) != 0:
                    self.classifier.load_state_dict(torch.load(pretrained_weights_path, map_location=get_device()))
            # only on CUDA due to: https://github.com/pytorch/pytorch/issues/125254
            self.classifier.compile() if torch.cuda.is_available() else None
            self.classifier.eval()
        self.is_initialized.set()

    def process(self, message: dict[str, any]):
        pose_buffer = message['pose_detection_results_batched']
        hoi_buffer = message['hoi_results_batched']
        detected_activities = message['activity_detection_results']

        graph_data_sequences = [
            self.classifier.create_graph(pose_results, hoi_results, detected_activities)
            for pose_results, hoi_results in zip(pose_buffer, hoi_buffer)
        ]

        if not self.classifier.training:
            with torch.no_grad():
                predictions = self.classifier(graph_data_sequences)
        else:
            predictions = self.classifier(graph_data_sequences)

        result = torch.flatten(predictions).cpu().item()
        self.classifier.velocities.clear()
        self.next(result)


class DimensionsSetter(MessageProcessor):

    def __init__(self, stream_app: CompositeBehaviouralClassifier):
        super().__init__()
        self.stream_app = stream_app

    def process(self, message: tuple[float, float]):
        self.stream_app.is_initialized.wait()
        self.stream_app.classifier.set_dimensions(message)
