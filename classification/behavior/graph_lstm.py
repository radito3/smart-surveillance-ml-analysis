import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Data

from classification.classifier import Classifier


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

        batch_sizes = 20
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


class GraphNetWithSAGPooling(torch.nn.Module):
    def __init__(self, node_features, hidden_dim):
        super(GraphNetWithSAGPooling, self).__init__()
        self.conv1 = GCNConv(node_features, 16)  # Experiment with 32 and maybe even 64 channels
        # Self-Attention Graph Pooling
        self.filter_pool = SAGPooling(16, ratio=0.8)  # Experiment with different retention ratios (0.6, 0.5)
        self.conv2 = GCNConv(16, hidden_dim)
        # TODO: try out global_max_pool as well
        self.global_pool = global_mean_pool  # Ensures a fixed-size output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        # TODO: is the global mean pooling necessary if we are passing the batch vector to the SAG pooling layer?
        x, edge_index, _, batch, _, _ = self.filter_pool(x, edge_index, None, batch)  # SAGPooling applied
        x = torch.relu(self.conv2(x, edge_index))
        print(f"batch: {batch}")
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

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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

    def classify_as_suspicious(self, vector: list[any]) -> bool:
        if len(self.buffer) < self.window_size:
            self.buffer.append(vector)
            return False

        graph_data_sequences = []
        for yolo_results in self.buffer:
            graph_data_sequences.append(self._create_graph(yolo_results))

        self.buffer = self.buffer[self.window_step:]

        predictions: torch.Tensor = self.forward(graph_data_sequences)
        print(predictions)
        # greater-than or equal
        result = torch.ge(predictions, 0.6).bool()
        return bool(result[0]) if len(result) > 0 else False

    def _create_graph(self, yolo_results):
        features, edges = self._extract_graph(yolo_results)
        return Data(x=torch.tensor(features, dtype=torch.float),
                    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous()
                    )

    def _extract_graph(self, yolo_results):
        """
        The node features should be:
         - orientation (where a person is facing)
         - velocity
         - average velocity?
         - body position - standing, sitting, laying | or body pose key points
         - activity - while very important, needs time series data, which won’t work for single frames
         - position in frame? (only matters in relation to other people)
         - whether he is holding something?

        The edges should have a higher index if people are close together multiplied by if they are facing each other.
        """

        # detections = np.random.randint(0, 255, (5, 4))  # [xmin, ymin, xmax, ymax] for 5 objects
        # format per tensor: [xmin, ymin, xmax, ymax]
        # we don't need the confidence scores here
        detections = [box_tensor for box_tensor, _ in yolo_results]

        # Calculate centroids for velocity calculations
        centroids_current = np.stack([(detections[:, 0] + detections[:, 2]) / 2,
                                      (detections[:, 1] + detections[:, 3]) / 2], axis=-1)

        # Velocity can be the magnitude of change in position
        # Direction can be encapsulated as the angle of movement
        if len(self.prev_detections) > 0:
            centroids_prev = np.stack([(self.prev_detections[:, 0] + self.prev_detections[:, 2]) / 2,
                                       (self.prev_detections[:, 1] + self.prev_detections[:, 3]) / 2], axis=-1)
            displacement = centroids_current - centroids_prev
            velocities = np.linalg.norm(displacement, axis=1)
            directions = np.arctan2(displacement[:, 1], displacement[:, 0])  # In radians
        else:
            velocities = np.zeros(len(centroids_current))
            directions = np.zeros(len(centroids_current))

        # Extract features for bounding boxes in xyxy format, velocity, and movement direction.
        features = np.hstack((detections, velocities.reshape(-1, 1), directions.reshape(-1, 1)))

        # Define edges based on proximity and relative direction
        threshold_distance = 50  # TODO: what unit of measurement is this? pixels?
        threshold_angle = np.pi / 4  # 45 degrees threshold
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.linalg.norm(centroids_current[i] - centroids_current[j])
                relative_angle = np.abs(directions[i] - directions[j])

                if distance < threshold_distance and relative_angle < threshold_angle:
                    edges.append((i, j))
                    edges.append((j, i))  # because the graph is undirected

        self.prev_detections = detections
        return features, edges
