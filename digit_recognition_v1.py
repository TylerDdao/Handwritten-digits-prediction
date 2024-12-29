import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import time


def preProcess(image, label):
    image = tf.image.resize(image, (28,28))
    image = tf.cast(image, tf.float32)/255.0

    if(image.shape[-1] == 1):
        image = tf.repeat(image, 3, axis=-1)
    
    return image, label

def trainWithMNISTDataset(model, nOfEpochs,histories):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train/255.0
    x_test = x_test/255.0

    # repeat the gray channel 3 times to simulate RGB 
    x_train = np.repeat(x_train, 3, axis=-1)
    x_test = np.repeat(x_test, 3, axis=-1)

    x_train = x_train.reshape(-1,28,28,3)
    x_test = x_test.reshape(-1,28,28,3)

    # y_train = to_categorical(y_train,10)
    # y_test = to_categorical(y_test,10)
    print("Start training with MNIST Dataset")
    MNISTHistory = model.fit(x_train,y_train,epochs = nOfEpochs, validation_data = (x_test,y_test))
    histories.append(MNISTHistory)
    return MNISTHistory

def trainWithSVHNDataset(model, nOfEpochs,histories):
    (ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',  # SVHN dataset in cropped format
    split=['train', 'test'],
    as_supervised=True,  # Loads data as (image, label)
    with_info=True  # Provides dataset metadata
    )
    
    batchSize =32
    trainDataset = ds_train.map(preProcess).batch(batchSize).shuffle(1000)
    testDataset = ds_test.map(preProcess).batch(batchSize).shuffle(1000)
    
    print("Start training with SVHN Dataset")
    SVHNHistory = model.fit(trainDataset, epochs = nOfEpochs, validation_data = testDataset)
    histories.append(SVHNHistory)
    return SVHNHistory

def trainWithEMNIST_Digits(model, nOfEpochs,histories):
    (ds_train, ds_test), ds_info = tfds.load(
        "emnist/digits",
        split=['train', 'test'],
        with_info=True,
        as_supervised=True
    )

    batchSize = 32
    trainDataset = ds_train.map(preProcess).batch(batchSize).shuffle(1000)
    testDataset = ds_test.map(preProcess).batch(batchSize).shuffle(1000)

    print("Start training with EMNIST/Digits Dataset")
    EMNIST_DigitsHistory = model.fit(trainDataset, epochs = nOfEpochs, validation_data = testDataset)
    histories.append(EMNIST_DigitsHistory)
    return EMNIST_DigitsHistory


def showGraph(histories):
    plt.figure(figsize=(10, 10))
    count = 0
    for h in histories:
        count = count+1
    # Accuracy plot
        plt.subplot(len(histories), 2, count)
        plt.plot(h.history['accuracy'], label='Train Accuracy')
        plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy_MNIST')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        count = count+1
        # Loss plot
        plt.subplot(len(histories), 2, count)
        plt.plot(h.history['loss'], label='Train Loss')
        plt.plot(h.history['val_loss'], label='Validation Loss')
        plt.title('Loss_MNIST')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    plt.show()

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Dropout(0.1),
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Dropout(0.1),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.2),
#     Dense(10, activation='softmax')
# ])

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(264, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
histories = []
startTime = time.time()
MNISTHistory = trainWithMNISTDataset(model, 20,histories)
# SVHNHistory = trainWithSVHNDataset(model, 20,histories)
# EMNIST_DigitsHistory = trainWithEMNIST_Digits(model,20,histories)
endTime = time.time()
trainingTime = endTime - startTime
print("Toltal training time: ", trainingTime, " seconds")
model.save('models\\model1_MNIST.h5')
showGraph(histories)