import logging
import sys
from threading import Thread

import cv2
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer, SubRegionExtractor
from analysis.human_object_interaction.interaction import HumanObjectInteractionAnalyzer
from classification.behavior.graph_lstm import CompositeBehaviouralClassifier, GraphBasedLSTMClassifier, DimensionsSetter
from analysis.object_detection.object_detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from messaging.message_broker import MessageBroker
from messaging.source.video_source_producer import VideoSourceProducer
from messaging.topology import Topology, KafkaStreams
from messaging.streams_builder import StreamsBuilder
from messaging.sink.training_sink import TrainingSink
from util.device import get_device


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        return video_path, label


def build_training_topology(broker: MessageBroker, fps: int, model: GraphBasedLSTMClassifier) -> Topology:
    window_size: int = 2 * fps
    window_step: int = fps // 2

    classifier = CompositeBehaviouralClassifier(node_features=13)
    classifier.inject_model(model)

    builder = StreamsBuilder(broker)

    builder.stream('video_source') \
        .named('pose-detection-app') \
        .process(PoseDetector()) \
        .to('pose_detection_results')

    builder.stream('video_source') \
        .named('object-detection-app') \
        .process(ObjectDetector()) \
        .to('object_detection_results')

    builder.stream('video_source') \
        .named('dimensions-setter-app') \
        .for_each(DimensionsSetter(classifier).process)

    builder.stream('video_source') \
        .join('pose_detection_results', lambda frame, poses: {'video_source': frame, 'pose_detection_results': poses}) \
        .named('activity-recognition-app') \
        .process(SubRegionExtractor()) \
        .window(size=window_size, step=window_step) \
        .process(MultiPersonActivityRecognitionAnalyzer()) \
        .to('activity_detection_results')

    builder.stream('object_detection_results') \
        .join('pose_detection_results', lambda objects, poses: {'object_detection_results': objects, 'pose_detection_results': poses}) \
        .named('human-object-interaction-app') \
        .process(HumanObjectInteractionAnalyzer()) \
        .window(size=window_size, step=window_step) \
        .to('hoi_results')

    builder.stream('pose_detection_results') \
        .window(size=window_size, step=window_step) \
        .join('activity_detection_results', lambda poses, activity: {'pose_detection_results': poses, 'activity_detection_results': activity}) \
        .join('hoi_results', lambda other, hoi: {**other, 'hoi_results': hoi}) \
        .named('graph-lstm-classifier-app') \
        .process(classifier) \
        .filter(lambda probability: probability != 0) \
        .to('output')

    return builder.build()


def get_video_fps(video_path: str) -> int:
    video_capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not video_capture.isOpened():
        return -1
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return int(fps)


def run_pipeline(video_path: str, model: GraphBasedLSTMClassifier) -> float:
    fps: int = get_video_fps(video_path)
    if fps == -1:
        return -1

    broker = MessageBroker()
    sink = TrainingSink()
    topology = build_training_topology(broker, fps, model)
    # do not bound the fps to not bottleneck the training time
    topology.add_source('video_source', VideoSourceProducer(broker, video_path, False))
    topology.add_sink('output', sink)

    streams_app = KafkaStreams(topology)
    streams_app.start()
    streams_app.wait()
    return sink.get_predicted_mean()


def train_model(model, train_loader, val_loader, epochs):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        true_labels = []
        predicted_probs = []

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = run_pipeline(inputs.item(), model)
            if outputs == -1:
                # video could not be opened
                continue

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            true_labels.append(labels.item())
            predicted_probs.append(outputs)

        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {running_loss/len(train_loader):.4f}')

        # there is no need to evaluate on every epoch, but we need it often due to the complexity of each epoch
        if epochs % 5 == 0 and epoch != 0:
            val_loss, recall, f1, roc_auc = evaluate_model(model, val_loader, criterion)

            print(f'Val Loss: {val_loss:.4f} - '
                  f'Recall: {recall:.4f} - F1 Score: {f1:.4f} - ROC AUC: {roc_auc:.4f}')

        torch.save(model.state_dict(), f'model_{2}_epoch_{epoch+1}.pth')


def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = run_pipeline(inputs.item(), model)
            if outputs == -1:
                # video could not be opened
                continue

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            true_labels.append(labels.item())
            predicted_probs.append(outputs)

    # Compute binary predictions from probabilities
    predicted_labels = [1 if prob > 0.6 else 0 for prob in predicted_probs]

    # Calculate metrics
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_probs)

    return running_loss / len(data_loader), recall, f1, roc_auc

if __name__ == '__main__':
    if len(sys.argv) < 3:
        logging.error("Invalid command-line arguments. Required <num_epochs> <model_config>")
        sys.exit(1)

    epochs = int(sys.argv[1])
    config = sys.argv[2]

    # TODO: load both the criminal activity videos and normal surveillance videos
    #  https://paperswithcode.com/dataset/ucf-crime
    #  https://roc-ng.github.io/XD-Violence/
    #  https://github.com/AlexanderMelde/SPHAR-Dataset
    #  for normal videos, check a public YouTube dataset?
    dataset = VideoDataset([], [])

    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.15 * len(dataset))   # 15% for validation
    test_size = len(dataset) - train_size - val_size  # 15% for testing

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # batch size of 1 because the data points are video URLs, and they need pre-processing before being passed to the model
    training_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    hyperparams = {
        'hidden_dim': [32, 64, 128],
        'pooling_channels': [32, 64, 128],
        'pooling_ratio': [0.8, 0.6],  # should be inversely-proportional on the crowd size - the larger the crowd, the lower the retention ratio
        'attention_heads': [2, 4],  # number of unique "viewpoints" the attention mechanism has on the data (in parallel)
        'lstm_layers': [1, 2]  # for each layer after 1, it would give the LSTM a more abstract view of the data (leaning towards crowd analysis)
    }
    # TODO: set the hyperparams based on the config
    model = GraphBasedLSTMClassifier(node_features=13).to(get_device())
    model.compile() if torch.cuda.is_available() else None

    train_model(model, training_loader, validation_loader, epochs)

    # TODO: run the model on the test_loader and output its final accuracy for the experiment
