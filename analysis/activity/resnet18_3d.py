import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms

from util.device import get_device


class ActivityRecognitionAnalyzer:

    def __init__(self):
        self.device = get_device()
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(self.device)
        self.model.compile() if torch.cuda.is_available() else None

    def predict_activity(self, window: list[cv2.typing.MatLike]) -> float:
        preprocessed_frames = self.__preprocess_frames(window)

        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).cpu().item()
        # divide by the total number of classes from the Kinetics-400 dataset (400) to normalize the value within the range [0, 1]
        return predicted_class / 399

    def __preprocess_frames(self, in_frames):
        # Convert frames to a NumPy array first and move them to the GPU due to a PyTorch warning that states:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        np_frames = np.array([cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2RGB) for frame in in_frames],
                             dtype=np.float32)
        frames = torch.from_numpy(np_frames).to(self.device) / 255.0  # Scale to [0, 1]
        # Permute to (T, C, Height, Width) where:
        # T is the temporal depth (number of frames)
        # C are the color channels (RGB in this case)
        frames = frames.permute(0, 3, 1, 2)
        normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        frames = normalize(frames)
        # Add batch dim and permute to (1, C, T, H, W)
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
        return frames
