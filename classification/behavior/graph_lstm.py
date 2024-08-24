import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGPooling
from torch_geometric.data import Data
from itertools import zip_longest

from analysis.types import AnalysisType
from classification.classifier import Classifier


class GraphNetWithSAGPooling(torch.nn.Module):
    # how will this inner network update its weights when training? where would the feedback come from?
    def __init__(self, node_features, hidden_dim):
        super(GraphNetWithSAGPooling, self).__init__()
        self.conv1 = GCNConv(node_features, 16)  # Experiment with 32 and maybe even 64 channels
        # Self-Attention Graph Pooling
        self.filter_pool = SAGPooling(16, ratio=0.8)  # Experiment with different retention ratios (0.6, 0.5)
        self.conv2 = GCNConv(16, hidden_dim)
        # TODO: try out global_max_pool as well
        self.global_pool = global_mean_pool  # Ensures a fixed-size output
        # self.global_pool = global_max_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        # TODO: is the global mean pooling necessary if we are passing the batch vector to the SAG pooling layer?
        x, edge_index, _, batch, _, _ = self.filter_pool(x, edge_index, None, batch)  # SAGPooling applied
        x = torch.relu(self.conv2(x, edge_index))
        # print(f"batch: {batch}")
        x = self.global_pool(x, batch)  # Global pool to enforce a fixed-sized output
        return x


class GraphBasedLSTMClassifier(torch.nn.Module, Classifier):
    def __init__(self,
                 node_features,  # Depending on the number of features per node (e.g. bbox + velocity + direction)
                 hidden_dim):  # Experiment with 16, 32 and 64
        torch.nn.Module.__init__(self)

        self.gnn = GraphNetWithSAGPooling(node_features, hidden_dim)
        # TODO: experiment with the following multi-layer LSTM architecture as well:
        #  nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Outputting a single probability

        self.window_size = 36  # 1.5 seconds of frames (at 24 fps)
        self.window_step = 10  # every 10 frames move to a new window
        self.prev_detections: list[torch.Tensor] = []
        self.buffer = []

    def forward(self, graph_data_sequences):
        # Assuming graph_data_sequences is a list of graph data for each time step
        embeddings = []
        for data in graph_data_sequences:
            graph_embedding = self.gnn(data)  # Obtain embedding for each graph
            embeddings.append(graph_embedding.unsqueeze(1))  # Add sequence dimension
        embeddings = torch.cat(embeddings, dim=1)  # Shape: (batch_size, sequence_length, hidden_dim)

        lstm_out, _ = self.lstm(embeddings)
        predictions = self.output_layer(lstm_out[:, -1, :])  # Classify based on the output of the last time step
        return torch.sigmoid(predictions)

    def classify_as_suspicious(self, dtype: AnalysisType, vector: list[any]) -> bool:
        if len(self.buffer) < self.window_size:
            self.buffer.append(vector)
            return False

        graph_data_sequences = [self._create_graph(yolo_results) for yolo_results in self.buffer]

        self.buffer = self.buffer[self.window_step:]

        predictions: torch.Tensor = self.forward(graph_data_sequences)
        print(f"predictions: {predictions}")
        # greater-than or equal
        result = torch.ge(torch.flatten(predictions), 0.5).bool()  # experiment and tune this threshold value
        return bool(result[0]) if len(result) > 0 else False

    def _create_graph(self, yolo_results):
        nodes, edges = self._extract_graph(yolo_results)
        return Data(x=torch.tensor(nodes, dtype=torch.float),
                    edge_index=torch.tensor([(x, y) for x, y, _ in edges], dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor([weight for _, _, weight in edges], dtype=torch.float)
                    )

    def _extract_graph(self, yolo_results):
        """
        The node features should be:
         - orientation (where a person is facing)
         - velocity
         - average velocity?
         - body position - standing, sitting, laying

         - activity - while very important, needs time series data, which won’t work for single frames
         - whether he is holding something?

        The edges should have a higher index if people are close together multiplied by if they are facing each other.
        """

        # we don't need the confidence scores here
        detections = [(box_tensor, box_id) for box_tensor, box_id, _ in yolo_results]
        if len(detections) < len(self.prev_detections):
            for _ in range(len(self.prev_detections) - len(detections)):
                detections.append((
                    torch.flatten(torch.full((1, 4), 0, dtype=torch.float32)),
                    -1
                ))

        # Calculate centroids for velocity calculations
        centroids_current = np.array([
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox, _ in detections
        ])

        # Velocity can be the magnitude of change in position
        # Direction can be encapsulated as the angle of movement
        if len(self.prev_detections) > 0:
            # centroids_prev = np.array([
            #     [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox, _ in self.prev_detections
            # ])

            # displacement = centroids_current - centroids_prev
            displacement = [self._get_centroid(current_bbox) - self._get_centroid(prev_bbox)
                            if current_id == prev_id else np.zeros(2, dtype=np.float32)
                            for (current_bbox, current_id), (prev_bbox, prev_id) in
                            # zip(detections, self.prev_detections)
                            zip_longest(detections, self.prev_detections, fillvalue=(np.zeros(4, dtype=np.float32), -1))
                            ]

            velocities = np.linalg.norm(displacement, axis=1)
            # directions = np.arctan2(displacement[:, 1], displacement[:, 0])  # In radians
            directions = np.array([np.arctan2(centroid[1], centroid[0]) for centroid in displacement])  # In radians
        else:
            velocities = np.zeros(len(centroids_current))
            directions = np.zeros(len(centroids_current))

        detects_only = [bbox for bbox, _ in detections]
        # Extract features for bounding boxes in xyxy format, velocity, and movement direction.
        features = np.hstack((detects_only, velocities.reshape(-1, 1), directions.reshape(-1, 1)))

        # Define edges based on proximity and relative direction
        # threshold_distance = 1000  # TODO: what unit of measurement is this? pixels?
        # threshold_angle = np.pi / 4  # 45 degrees threshold
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.linalg.norm(centroids_current[i] - centroids_current[j])
                relative_angle = np.abs(directions[i] - directions[j])
                edge_weight = distance / relative_angle if relative_angle != 0 else distance

                # is a complete (fully-connected) graph good here? why not?
                edges.append((i, j, edge_weight))
                edges.append((j, i, edge_weight))  # because the graph is undirected

        self.prev_detections = detections
        return features, edges

    @staticmethod
    def _get_centroid(bbox) -> np.ndarray:
        # bounding box in xyxy format [xmin, ymin, xmax, ymax]
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
