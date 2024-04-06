import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 11
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.5


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    if not labels:
        sys.exit("No labels found. Check your dataset directory.")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    data = []

    #making sure we're in the right directory,
    print(f" current directory is {os.getcwd()}, loading data from {data_dir}")
    # for every part of directory, specifically filenames and folder
    for directory_path, _, filenames in os.walk(data_dir):
        print(directory_path)
        for filename in filenames:
            #find image path,
            image_path = os.path.join(directory_path, filename)
            #read image, convert to value array
            image = cv2.imread(str(image_path))
            #resize image to width and height
            resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            #normalize to reduce load
            normalized_image = np.array(resized, dtype="float32") / 255.0
            #get label from the directory path
            label = int(os.path.basename(directory_path))
            #append to data dictionary
            data.append({"image": normalized_image, "label": label})

    images = [row["image"] for row in data]
    labels = [row["label"] for row in data]

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    """The get_model function should return a compiled neural network model.
    You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
    The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
    The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
    different numbers of convolutional and pooling layers
    different numbers and sizes of filters for convolutional layers
    different pool sizes for pooling layers
    different numbers and sizes of hidden layers
    dropout"""
    model = tf.keras.models.Sequential([
        #convolution layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        #max-Pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #convolutional layer 2: electric boogaloo
        tf.keras.layers.Conv2D(250, (4, 4), activation="relu"),
        # Max-pooling layer 2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #lowkey dk what this does, flatten layers?
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
#with open(traffic_model.keras) as model:
#     url = input("Enter the image address")
#     image = cv2.imread(str(url))
#     resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

if __name__ == "__main__":
    main()

