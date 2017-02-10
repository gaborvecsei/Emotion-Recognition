from collections import Counter

import numpy as np
import pandas as pd

from utils import preprocessImage, flipImage, splitDataRandomly

df = pd.read_csv('data/fer2013.csv', header=0)

labels = list(df['emotion'])
print("Label frequencies: {0}".format(dict(Counter(labels))))

X = []
y = []

for i in range(df.shape[0]):
    img_pixels = df['pixels'][i]
    label = df['emotion'][i]

    img_pixels = img_pixels.split(' ')
    img_pixels = np.array(img_pixels, np.uint8)
    img = np.reshape(img_pixels, (48, 48))

    X.append(img)
    y.append(label)

    if i % 1000 == 0:
        print("{0} rows done!".format(i))

X = np.array(X)
y = np.array(y)

# Split the data: training - testing
X_train, y_train, X_test, y_test = splitDataRandomly(X, y, 0.9)

print("X_train shape: {0}\ny_train shape: {1}".format(X_train.shape, y_train.shape))
print("X_test shape: {0}\ny_test shape: {1}".format(X_test.shape, y_test.shape))

print("Train label frequencies: {0}".format(dict(Counter(y_train))))
print("Test label frequencies: {0}".format(dict(Counter(y_test))))

np.save("data/X_train.npy", X_train)
np.save("data/Y_train.npy", y_train)

np.save("data/X_test.npy", X_test)
np.save("data/Y_test.npy", y_test)

print("Arrays are saved!")
