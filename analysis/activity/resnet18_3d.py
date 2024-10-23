import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights

from util.device import get_device


class ActivityRecognitionAnalyzer:

    def __init__(self):
        self.device = get_device()
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(self.device)
        self.model.compile() if torch.cuda.is_available() else None

    def predict_activity(self, window: list[cv2.typing.MatLike]) -> int:
        preprocessed_frames = self.__preprocess_frames(window)

        with torch.no_grad():
            outputs = self.model(preprocessed_frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).cpu().item())

        return predicted_class

    def __preprocess_frames(self, in_frames, input_size=(112, 112)):
        # Convert frames to a NumPy array first and move them to the GPU due to a PyTorch warning that states:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        np_frames = np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), input_size) for frame in in_frames],
                             dtype=np.float32)
        frames = torch.from_numpy(np_frames).to(self.device)

        # Normalize the frames (operating on GPU tensors)
        frames /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)
        frames = (frames - mean) / std

        # Convert HWC to CHW by permuting dimensions and adding batch dimension
        frames = frames.permute(0, 3, 1, 2)  # From (T, H, W, C) to (T, C, H, W)

        # Add a batch dimension
        frames = frames.unsqueeze(0)  # Shape: [1, T, C, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)  # Shape: [1, C, T, H, W]
        return frames
