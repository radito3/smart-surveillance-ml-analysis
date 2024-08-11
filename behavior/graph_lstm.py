import cv2
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from analysis.classifier import Classifier


def extract_graph(frame, prev_detections):
    """Extract features for bounding boxes in xyxy format, velocity, and movement direction."""
    # Simulated object detection in format [xmin, ymin, xmax, ymax]
    detections = np.random.randint(0, 255, (5, 4))  # [xmin, ymin, xmax, ymax] for 5 objects

    # Calculate centroids for velocity calculations
    centroids_current = np.stack([(detections[:, 0] + detections[:, 2]) / 2,
                                  (detections[:, 1] + detections[:, 3]) / 2], axis=-1)

    # Velocity can be the magnitude of change in position
    # Direction can be encapsulated as the angle of movement
    if prev_detections is not None:
        centroids_prev = np.stack([(prev_detections[:, 0] + prev_detections[:, 2]) / 2,
                                   (prev_detections[:, 1] + prev_detections[:, 3]) / 2], axis=-1)
        displacement = centroids_current - centroids_prev
        velocities = np.linalg.norm(displacement, axis=1)
        directions = np.arctan2(displacement[:, 1], displacement[:, 0])  # In radians
    else:
        velocities = np.zeros(len(centroids_current))
        directions = np.zeros(len(centroids_current))

    features = np.hstack((detections, velocities.reshape(-1, 1), directions.reshape(-1, 1)))

    # Define edges based on proximity and relative direction
    threshold_distance = 50
    threshold_angle = np.pi / 4  # 45 degrees threshold
    edges = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distance = np.linalg.norm(centroids_current[i] - centroids_current[j])
            relative_angle = np.abs(directions[i] - directions[j])

            if distance < threshold_distance and relative_angle < threshold_angle:
                edges.append((i, j))
                edges.append((j, i))  # because the graph is undirected

    return features, edges, detections


def create_graph(frame, prev_detections):
    features, edges, detections = extract_graph(frame, prev_detections)
    prev_detections = detections  # update this when buffering
    return Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous()
                )


class GraphFeatureLSTM(nn.Module):
    def __init__(self, input_dim,  # Depending on the number of features per node (bbox + velocity + direction)
                 hidden_dim,
                 num_layers):
        super(GraphFeatureLSTM, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Outputting a single probability
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x, batch_index):
        # x: Node features [batch_size, num_nodes, num_features]
        # batch_index: To indicate graph structure when nodes from several graphs are batched together
        num_graphs = batch_index.max().item() + 1

        # Embedding layer
        x = self.embedding(x)

        # Pack sequence for LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(x, batch_sizes, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # Unpack the sequence
        output, input_sizes = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Pooling (Global mean pooling in this case)
        pooled_output = global_mean_pool(output, batch_index)

        # Output layer for each graph with sigmoid activation
        out = torch.sigmoid(self.output_layer(pooled_output))
        return out


class GraphBasedLSTMClassifier(torch.nn.Module, Classifier):
    def __init__(self, node_features, hidden_dim):
        super().__init__()
        self.gnn = GraphNetWithSAGPooling(node_features, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # TODO: ensure the activation function is a sigmoid
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.in_training = True
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, graph_data_sequences):
        if self.in_training is False:
            self.train()
            self.in_training = True

        # Assuming graph_data_sequences is a list of graph data for each time step
        embeddings = []
        for data in graph_data_sequences:
            graph_embedding = self.gnn(data)  # Obtain embedding for each graph
            embeddings.append(graph_embedding.unsqueeze(1))  # Add sequence dimension
        embeddings = torch.cat(embeddings, dim=1)  # Shape: (batch_size, sequence_length, hidden_dim)

        lstm_out, _ = self.lstm(embeddings)
        predictions = self.classifier(lstm_out[:, -1, :])  # Classify based on the output of the last time step
        return predictions

    def classify_as_suspicious(self, vector: list[any]) -> bool:
        if self.in_training:
            self.eval()
            self.in_training = False

        # TODO: accumulate frames with a sliding window and pass them in bulk
        predictions: torch.Tensor = self.forward(vector)
        # greater-than or equal
        return torch.ge(predictions, 0.6).bool()


class GraphNetWithSAGPooling(torch.nn.Module):
    def __init__(self, node_features, hidden_dim):
        super(GraphNetWithSAGPooling, self).__init__()
        self.conv1 = GCNConv(node_features, 16)
        # Self-Attention Graph Pooling
        self.pool1 = SAGPooling(16, ratio=0.8)  # TODO: experiment with different retention ratios (0.6, 0.5)
        self.conv2 = GCNConv(16, hidden_dim)
        self.global_pool = global_mean_pool  # Ensures a fixed-size output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        # TODO: is the global mean pooling necessary if we are passing the batch vector to the SAG pooling layer?
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # SAGPooling applied
        x = torch.relu(self.conv2(x, edge_index))
        x = self.global_pool(x, batch)  # Global pool to enforce a fixed-sized output
        return x
