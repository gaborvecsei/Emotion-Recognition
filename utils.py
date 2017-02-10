import cv2
import numpy as np
import seaborn as sns

emotionDict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}


def normalizeArray(array):
    array = array.astype(np.float)
    array /= 255.0
    return array


def preprocessImage(image):
    tmpImg = cv2.GaussianBlur(image, (3, 3), 5)
    img = cv2.addWeighted(image, 1.5, tmpImg, -0.5, 0)
    img = cv2.equalizeHist(img)
    return img


def flipImage(image):
    # Horizontal flip
    return cv2.flip(image, 1)


def splitDataRandomly(X, y, percentage):
    s = X.shape
    mask = np.random.rand(s[0]) <= percentage
    X_train = X[mask]
    y_train = y[mask]
    X_test = X[~mask]
    y_test = y[~mask]
    return X_train, y_train, X_test, y_test


def findBiggerSqrtNumber(num):
    """
    The square of the returned number great or equal to the original number

    :param num: number (integer)
    :return: sqrt number
    """

    tmpPos = num
    while np.sqrt(tmpPos) % 1 != 0:
        tmpPos += 1
    return int(np.sqrt(tmpPos))


def plotPrediction(prediction_confidences):
    """
    Plots the prediction than encodes it to base64
    :param pred: prediction accuracies
    :return: base64 encoded image as string
    """

    labels = list(emotionDict.values())
    sns.set_context(rc={"figure.figsize": (4, 3)})
    with sns.color_palette("RdBu_r", 3):
        ax = sns.barplot(x=labels, y=prediction_confidences)
    ax.set_xticklabels(labels=labels, rotation=30)
    ax.set(ylim=(0, 1))
    return ax
