import numpy as np
import cv2
import random


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


def dataGenerator(batch_size, X, y):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            randomIndex = random.randint(0, len(X) - 1)
            label = y[randomIndex]
            image = X[randomIndex]
            batch_X.append(image)
            batch_y.append(label)
        yield np.array(batch_X), np.array(batch_y)
