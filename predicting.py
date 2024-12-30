import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


modelName = "model1_MNIST.h5"
# Load the trained model
model = tf.keras.models.load_model(modelName) # best model for image based classifier

# Step 10: Make predictions on a new image
def prepare_imageRGB(image_path, target_size=(28, 28)):
    # Open the image and convert to RGB (3 channels)
    img = Image.open(image_path).convert('RGB')  # Convert image to RGB

    # Resize the image to the target size (32, 32)
    img_resized = img.resize(target_size)

    # Convert image to a NumPy array
    img_array = np.array(img_resized)

    # Check the mean intensity for each channel
    mean_intensity = np.mean(img_array)

    # If the image is bright (likely black on white), invert it
    if mean_intensity > 127:  # Threshold can be adjusted based on testing
        img_array = cv2.bitwise_not(img_array)

    # Normalize the pixel values to [0, 1]
    img_array = img_array / 255.0

    # The image is now in RGB format, so we don't need to add a channel dimension

    # Display the processed image
    # plt.imshow(img_array, cmap='gray')
    # plt.title("Processed Image")
    # plt.axis('off')
    # plt.show()

    # We add the batch dimension (1, 28, 28, 3)
    return np.expand_dims(img_array, axis=0)

# def prepare_image(image_path, target_size=(28, 28)):
#     img = Image.open(image_path).convert('L')  # Convert image to grayscale

#     img_resized = img.resize(target_size)  # Resize image to 28x28
#     img_array = np.array(img_resized)  # Convert image to a NumPy array

#     # Check the mean intensity
#     mean_intensity = np.mean(img)

#     # If the image is bright (likely black on white), invert it
#     if mean_intensity > 127:  # Threshold can be adjusted based on testing
#         img_array = cv2.bitwise_not(img_array)

#     img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
#     img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (28, 28, 1)

#     # Display the processed image
#     # plt.imshow(img_array, cmap='gray')
#     # plt.title("Processed Image")
#     # plt.axis('off')
#     # plt.show()
#     return np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 28, 28, 1)

# # Example prediction: Use the model to predict a new image
# image_path = 'testv4\\6.png'  # Replace with your image path
# image_data = prepare_image(image_path)
# prediction = model.predict(image_data)
# predicted_class = np.argmax(prediction)

# print(f"Predicted: {predicted_class}")
# print("Expected: ", image_path)

probability =[]
results =[]

for i in range(0,10):
    testname = 'testv3\\test'
    testname = testname + str(i)
    testname = testname + '.JPG'
    image_path = testname  # Replace with your image path
    image_data = prepare_imageRGB(image_path)
    prediction = model.predict(image_data)
    predicted_class = np.argmax(prediction)
    results.append(predicted_class)
    probability.append(prediction)

for i in range(0,10):
    # print("Probability: ", probability)
    print("Result: ", results[i])
    print("Expected: ", i)

# image_path = 'testv4\\13.png'  # Replace with your image path
# image_data = prepare_image(image_path)
# prediction = model.predict(image_data)
# predicted_class = np.argmax(prediction)
# print(predicted_class)
# print("Expected: ", image_path)
