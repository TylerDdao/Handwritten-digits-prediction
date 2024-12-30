import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load the model
model = tf.keras.models.load_model("model1_MNIST.h5")

def prepare_imageRGB(image_path, target_size=(28, 28)):
    # Open the image and convert to RGB (3 channels)
    img = Image.open(image_path).convert('RGB')  # Convert image to RGB

    # Resize the image to the target size (28, 28)
    img_resized = img.resize(target_size)

    # Convert image to a NumPy array
    img_array = np.array(img_resized)

    # Check the mean intensity for each channel
    mean_intensity = np.mean(img_array)

    # If the image is bright (likely black on white), invert it
    if mean_intensity > 127:
        img_array = cv2.bitwise_not(img_array)

    # Normalize the pixel values to [0, 1]
    img_array = img_array / 255.0

    # Add the batch dimension (1, 28, 28, 3)
    return np.expand_dims(img_array, axis=0)

workFile = 'test\\test0.JPG' # Replace by the relative path to your image you want to test on
image_path = workFile  # Replace with your image path
image_data = prepare_imageRGB(image_path)
prediction = model.predict(image_data)
predicted_class = np.argmax(prediction)
print(predicted_class)
