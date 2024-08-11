import torch
from datetime import timedelta
import torchvision.transforms as transforms
import cv2
import csv
from torchvision.models.video import r3d_18

from analysis.types import AnalysisType
from analysis.video_buffer_analyzer import VideoBufferAnalyzer


class ActivityRecognitionAnalyzer(VideoBufferAnalyzer):

    def __init__(self, fps: int, window_size: timedelta):
        super().__init__(fps, window_size)
        self.model = r3d_18(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        # Load the class labels
        with open('../kinetics_400_labels.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.kinetics_classes = [name for _, name in reader]

    def analysis_type(self) -> AnalysisType:
        return AnalysisType.ActivityDetection

    def analyze_video_window(self, window: list[cv2.typing.MatLike]) -> list[any]:
        preprocessed_frames = self.__preprocess_frames(window)
        preprocessed_frames = preprocessed_frames.to(self.device)
        # Make predictions
        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        predicted_activity = self.kinetics_classes[predicted_class.item()]
        return [0, 0, predicted_activity, 0, 0]  # TODO: one-hot encode supported activities

    def __preprocess_frames(self, frames: list[cv2.typing.MatLike]) -> torch.Tensor:
        # Convert frames from BGR to RGB
        preprocessed_frames = [self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        preprocessed_frames = torch.stack(preprocessed_frames).unsqueeze(0)  # Add batch dimension
        return preprocessed_frames
