import random

import numpy as np


def dataGenerator(batch_size, X, y):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            randomIndex = random.randint(0, len(X) - 1)
            image = X[randomIndex]
            label = y[randomIndex]
            batch_X.append(image)
            batch_y.append(label)
        yield np.array(batch_X), np.array(batch_y)
