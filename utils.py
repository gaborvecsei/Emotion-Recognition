import cv2
import numpy as np
import seaborn as sns

EMOTION_DICT = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}


def normalize_array(image_array):
    """
    Normalize and image array. (values between 0...1)
    :param array: image array
    :return: normalized array
    """

    array = image_array.astype(np.float)
    array /= 255.0
    return array


def preprocess_image(image):
    """
    Sharpens the image, then applies histogram equalization
    """

    tmp_img = cv2.GaussianBlur(image, (3, 3), 5)
    img = cv2.addWeighted(image, 1.5, tmp_img, -0.5, 0)
    img = cv2.equalizeHist(img)
    return img


def flip_image(image):
    """
    Flips the image horizontally
    """

    return cv2.flip(image, 1)


def split_data_randomly(X, y, percentage):
    """
    Splits the data to training and testing
    """

    s = X.shape
    mask = np.random.rand(s[0]) <= percentage
    X_train = X[mask]
    y_train = y[mask]
    X_test = X[~mask]
    y_test = y[~mask]
    return X_train, y_train, X_test, y_test


def find_bigger_sqrt_number(num):
    """
    The square of the returned number great or equal to the original number

    :param num: number (integer)
    :return: sqrt number
    """

    tmpPos = num
    while np.sqrt(tmpPos) % 1 != 0:
        tmpPos += 1
    return int(np.sqrt(tmpPos))
