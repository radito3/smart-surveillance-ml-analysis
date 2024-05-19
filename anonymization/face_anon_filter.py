import cv2
import os
from mtcnn import MTCNN
import tensorflow as tf

# Set TensorFlow logging level to reduce verbosity
# FIXME: doesn't work... it still outputs too much logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class FaceAnonymizer:
    def __init__(self):
        # Initialize MTCNN face detector
        self.detector = MTCNN()
        # State variable to store current bounding boxes
        self.current_faces = []

    # FIXME: this still drops the rectangles sometimes, leading to periods of non-anonymized faces
    def update_faces(self, new_faces):
        if new_faces and len(new_faces) > 0:
            self.current_faces = new_faces

    @staticmethod
    def blur_face(image, x, y, w, h):
        # Extract the region of interest (face)
        roi = image[y:y+h, x:x+w]
        # Apply Gaussian blur to the face
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        # Replace the face in the image with the blurred face
        image[y:y+h, x:x+w] = blurred_roi
        return image

    @staticmethod
    def pixelate_face(image, x, y, w, h):
        # Extract the region of interest (face)
        roi = image[y:y+h, x:x+w]
        # Define the size of the pixelated effect
        height, width = roi.shape[:2]
        temp = cv2.resize(roi, (width // 10, height // 10), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        # Replace the face in the image with the pixelated face
        image[y:y+h, x:x+w] = pixelated_roi
        return image

    def __call__(self, image, effect='blur'):
        # Detect faces in the image
        results = self.detector.detect_faces(image)

        # Extract bounding boxes from results
        new_faces = [result['box'] for result in results]

        # Update current faces if new faces are detected
        self.update_faces(new_faces)

        # Draw rectangles around the current faces
        for (x, y, w, h) in self.current_faces:
            if effect == 'blur':
                image = self.blur_face(image, x, y, w, h)
            elif effect == 'pixelate':
                image = self.pixelate_face(image, x, y, w, h)

        return image
