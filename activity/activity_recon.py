import torch
import torchvision.transforms as transforms
import cv2
# import numpy as np
from torchvision.models.video import r3d_18

from analysis.video_buffer_analyzer import VideoBufferAnalyzer


# Function to extract frames from the video
# def extract_frames(video_path, num_frames=16):
#     # helpful: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4029702/
#
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
#
#     for i in range(total_frames):
#         ret, frame = cap.read()
#         if i in frame_indices:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#             frames.append(frame)
#         if len(frames) == num_frames:
#             break
#
#     cap.release()
#     return frames


class ActivityRecognitionAnalyzer(VideoBufferAnalyzer):

    def __init__(self, window_size: int, fps: int):
        self.num_frames = window_size * fps
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
        # TODO: ensure file is closed after reading
        # Load the class labels
        self.kinetics_classes = [line.strip() for line in open("path_to_kinetics_class_labels.txt")]

    def analyze(self, video_window: list[cv2.typing.MatLike], *args, **kwargs) -> list[int]:
        preprocessed_frames = self.__preprocess_frames(video_window)
        preprocessed_frames = preprocessed_frames.to(self.device)
        # Make predictions
        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        predicted_activity = self.kinetics_classes[predicted_class.item()]
        return [0, 0, predicted_activity, 0, 0]  # TODO: one-hot encode supported activities

    def get_num_frames(self) -> int:
        return self.num_frames

    def __preprocess_frames(self, frames: list[cv2.typing.MatLike]) -> torch.Tensor:
        # TODO: Convert frames from BGR to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed_frames = [self.transform(frame) for frame in frames]
        preprocessed_frames = torch.stack(preprocessed_frames).unsqueeze(0)  # Add batch dimension
        return preprocessed_frames
