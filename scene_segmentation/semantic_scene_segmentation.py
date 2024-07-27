import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(image_path, target_size=(513, 513)):
    """ Load an image using OpenCV and resize it to target size """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    return image


def prepare_image(image):
    """ Preprocess the image for the DeepLabV3 model """
    img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    return img


def create_deeplab_model():
    """ Initialize a DeepLabV3 model with a MobileNetV2 backbone """
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

    # Using the output from the last layer before final classifier
    last_layer = model.get_layer('block_16_project_BN').output
    x = tf.keras.layers.Conv2D(1, (1, 1), name='conv_logits')(last_layer)
    x = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear', name='upsampling')(x)

    # Final model
    deeplab_model = tf.keras.Model(inputs=model.input, outputs=x)
    return deeplab_model


def segment_image(image_path):
    """ Segment the image into classes """
    # Load and prepare the model
    deeplab_model = create_deeplab_model()

    # Load and prepare the image
    image = load_image(image_path)
    preprocessed_image = prepare_image(image)

    # Get the labeled image from model prediction
    predictions = deeplab_model.predict(preprocessed_image)
    label_image = np.argmax(predictions[0], axis=-1)
    label_image = np.squeeze(label_image)

    return image, label_image


def plot_results(image, label_image):
    """ Plot the original image and segmented output """
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Segmented Image')
    plt.imshow(label_image)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    image_path = 'your_image_here.jpg' # Change this to your image path
    image, label_image = segment_image(image_path)
    plot_results(image, label_image)
