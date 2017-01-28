import numpy as np
import cv2


def preprocessImage(image):
    tmpImg = cv2.GaussianBlur(image, (3, 3), 5)
    img = cv2.addWeighted(image, 1.5, tmpImg, -0.5, 0)
    img = cv2.equalizeHist(img)
    img = img.astype(np.float)
    img /= 255.0
    return img


def flipImage(image):
    # Horizontal flip
    return cv2.flip(image, 1)

def splitData(X,y,percentage):
    s = X.shape
    mask = np.random.rand(s[0]) <= percentage
    X_train = X[mask]
    y_train = y[mask]
    X_test = X[~mask]
    y_test = y[~mask]
    return X_train, y_train, X_test, y_test