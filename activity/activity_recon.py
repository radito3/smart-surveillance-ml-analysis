import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.models.video import r3d_18

# Load the pre-trained R3D-18 model
model = r3d_18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 171)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])


# Function to extract frames from the video
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()
    return frames


# Function to preprocess frames
def preprocess_frames(frames, transform):
    preprocessed_frames = [transform(frame) for frame in frames]
    preprocessed_frames = torch.stack(preprocessed_frames).unsqueeze(0)  # Add batch dimension
    return preprocessed_frames


# Load and preprocess the video frames
video_path = 'path_to_your_video.mp4'
frames = extract_frames(video_path)
preprocessed_frames = preprocess_frames(frames, transform)
preprocessed_frames = preprocessed_frames.to(device)

# Make predictions
with torch.no_grad():
    outputs = model(preprocessed_frames)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

# Load the class labels
kinetics_classes = [line.strip() for line in open("path_to_kinetics_class_labels.txt")]

# Print the predicted activity
predicted_activity = kinetics_classes[predicted_class.item()]
print(f'Predicted activity: {predicted_activity}')
