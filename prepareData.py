from collections import Counter

import numpy as np
import pandas as pd

from utils import preprocessImage, flipImage

# If it is true than the images are flipped horizontally to enrich data
DATA_AUGMENTATION = True

df = pd.read_csv('data/fer2013.csv', header=0)

labels = list(df['emotion'])
print ("Label frequencies: {0}".format(dict(Counter(labels))))

X = []
Y = []

for i in range(df.shape[0]):
    img_pixels = df['pixels'][i]
    label = df['emotion'][i]

    img_pixels = img_pixels.split(' ')
    img_pixels = np.array(img_pixels, np.uint8)
    img = np.reshape(img_pixels, (48, 48))

    processedImage = preprocessImage(img)
    extendedImg = np.reshape(processedImage, (1, 48, 48))

    X.append(extendedImg)
    Y.append(label)

    if DATA_AUGMENTATION:
        flippedImg = flipImage(processedImage)
        extendedFlippedImg = np.reshape(flippedImg, (1, 48, 48))

        X.append(extendedFlippedImg)
        Y.append(label)

    if i % 1000 == 0:
        print ("{0} rows done!".format(i))

X = np.array(X)
Y = np.array(Y)

print ("X shape: {0}\nY shape: {1}".format(X.shape, Y.shape))

np.save("data/X_train.npy", X)
np.save("data/Y_train.npy", Y)

print ("Arrays are saved!")
