import torch
from datetime import timedelta
import cv2
import csv
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights

from analysis.types import AnalysisType
from analysis.video_buffer_analyzer import VideoBufferAnalyzer


class ActivityRecognitionAnalyzer(VideoBufferAnalyzer):

    def __init__(self, fps: int, window_size: timedelta, window_step: int):
        super().__init__(fps, window_size, window_step)
        self.model = None
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Load the class labels
        with open('analysis/activity/kinetics_400_labels.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.kinetics_classes = [row['name'] for row in reader]

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.ActivityDetection

    def analyze_video_window(self, window: list[cv2.typing.MatLike]) -> list[any]:
        if self.model is None:
            self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            self.model.to(self.device)

        preprocessed_frames = self.__preprocess_frames(window)

        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # should this be copied to CPU memory?
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return [predicted_class]

    def __preprocess_frames_cpu(self, frames, input_size=(112, 112)):
        # Pre-allocate a NumPy array for all frames in CHW format
        num_frames = len(frames)
        processed_frames = np.empty((num_frames, 3, input_size[0], input_size[1]), dtype=np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for i, frame in enumerate(frames):
            # Convert frames from BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            resized_frame = cv2.resize(rgb_frame, input_size)
            # Normalize the frame
            resized_frame = resized_frame / 255.0
            resized_frame = (resized_frame - mean) / std
            # Convert to CHW format (HWC to CHW)
            processed_frames[i] = np.transpose(resized_frame, (2, 0, 1))

        # Convert the processed frames to a tensor and add batch dimension
        return torch.tensor(processed_frames).unsqueeze(0)  # Shape: [1, T, C, H, W]

    def __preprocess_frames(self, in_frames, input_size=(112, 112)):
        # Convert frames to a NumPy array and move them to the GPU
        np_frames = np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), input_size) for frame in in_frames],
                             dtype=np.float32)
        frames: torch.Tensor = torch.tensor(np_frames).to(self.device)

        # Normalize the frames (operating on GPU tensors)
        frames /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)
        frames = (frames - mean) / std

        # Convert HWC to CHW by permuting dimensions and adding batch dimension
        frames = frames.permute(0, 3, 1, 2)  # From (T, H, W, C) to (T, C, H, W)

        # Add a batch dimension and return the tensor
        frames = frames.unsqueeze(0)  # Shape: [1, T, C, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)  # Shape: [1, C, T, H, W]
        return frames
