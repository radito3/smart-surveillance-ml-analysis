import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
# model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

# Define the image transformation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to get the segmentation mask
def get_segmentation_mask(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_predictions = output.argmax(0)
    return output_predictions.byte().cpu().numpy()

# Function to overlay mask on image
def overlay_mask_on_image(image, mask, alpha=0.6):
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
        [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
        [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0]
    ])

    colored_mask = colors[mask]
    overlayed_image = cv2.addWeighted(image, alpha, colored_mask, 1 - alpha, 0)
    return overlayed_image

# Load and preprocess the input image
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the segmentation mask
segmentation_mask = get_segmentation_mask(image)

# Overlay the mask on the original image
overlayed_image = overlay_mask_on_image(image, segmentation_mask)

# Display the original image and the segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlayed_image)
plt.title("Segmented Image")
plt.axis('off')

plt.show()
