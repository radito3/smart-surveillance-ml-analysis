# create all models, model variations with different neurons per layer, different number of layers, etc.
# download criminal activity datasets
# find and download surveillance footage of normal behaviour
# split 70% training, 15% validation, and 15% test
# create processes for each model
# train 100 epochs, check accuracy, repeat training until the difference in accuracy between validation iterations
#    is less than a threshold

import time
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from analysis.activity.multi_person_activity_recon import MultiPersonActivityRecognitionAnalyzer
from analysis.analyzer_with_cache_aside import CacheAsideAnalyzer
from analysis.object_detection.detector import ObjectDetector
from analysis.pose_detection.pose_detector import PoseDetector
from classification.behavior.graph_lstm import GraphBasedLSTMClassifier
from classification.classifier import Classifier

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True)


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


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

hyperparams = {
    'hidden_dim': [16, 32, 64],
    'pooling_channels': [16, 32, 64],
    'pooling_ratio': [0.8, 0.6],
    'global_pool_type': ['mean', 'max'],
    'lstm_layers': [1, 2]
}

model = GraphBasedLSTMClassifier(node_features=5, window_size=48, window_step=12)
model.train()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()


def run_pipeline(video_url: str, classifier: Classifier) -> float:
    video_source = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    assert video_source.isOpened(), "Error opening video"
    target_fps: int = 24
    target_frame_interval: float = 1./target_fps
    prev_timestamp: float = 0

    cacheable_people_detector = CacheAsideAnalyzer(ObjectDetector(), cache_life=1)
    analyzers = [cacheable_people_detector, PoseDetector(),
                 MultiPersonActivityRecognitionAnalyzer(cacheable_people_detector, target_fps,
                                                        timedelta(seconds=2), target_fps // 2)]

    predictions: list[float] = []
    while video_source.isOpened():
        time_elapsed: float = time.time() - prev_timestamp
        ok, frame = video_source.read()
        if not ok:
            break

        if time_elapsed > target_frame_interval:
            prev_timestamp = time.time()
            # sequential processing will be very slow, but it's okay for training
            results = [(analyzer.analysis_type(), analyzer.analyze(frame)) for analyzer in analyzers]
            for dtype, data in results:
                conf = classifier.classify_as_suspicious(dtype, data)
                if conf != 0:
                    predictions.append(conf)
    video_source.release()
    return np.mean(predictions)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = run_pipeline(inputs, model)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()
        # Assuming you have true labels (y_true) and predicted probabilities (y_pred_probs)
        y_pred = (outputs >= 0.5).astype(int)

        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, outputs)

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

# Perform k-fold cross-validation to ensure your model generalizes well across different
# scenes and configurations

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       { 'Training' : avg_loss, 'Validation' : avg_vloss },
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}.pt'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# to load a saved version of the model
saved_model.load_state_dict(torch.load(PATH))
