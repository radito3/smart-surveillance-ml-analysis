import torch
from datetime import timedelta
import torchvision.transforms as transforms
import cv2
import csv
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights

from analysis.types import AnalysisType
from analysis.video_buffer_analyzer import VideoBufferAnalyzer


class ActivityRecognitionAnalyzer(VideoBufferAnalyzer):

    def __init__(self, fps: int, window_size: timedelta, window_step: int):
        super().__init__(fps, window_size, window_step)
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.model.eval()  # Set the model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            # transforms.Resize((128, 171)),
            transforms.Resize((112, 112)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load the class labels
        with open('analysis/activity/kinetics_400_labels.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.kinetics_classes = [row['name'] for row in reader]

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.ActivityDetection

    def analyze_video_window(self, window: list[cv2.typing.MatLike]) -> list[any]:
        preprocessed_frames = self.preprocess_frames_v3(window)
        preprocessed_frames = preprocessed_frames.to(self.device)

        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()

        return [predicted_class]

    def preprocess_frames(self, frames, input_size=(112, 112)):
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

    def preprocess_frames_v3(self, frames, input_size=(112, 112)):
        # Convert frames to a NumPy array and move them to the GPU
        frames = np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), input_size) for frame in frames],
                          dtype=np.float32)
        frames = torch.tensor(frames).to(self.device)

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
