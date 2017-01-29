import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

from utils import preprocessImage, emotionDict, findBiggerSqrtNumber

K.set_image_dim_ordering('th')

# Magic...
tf.python.control_flow_ops = tf

SAVE_MODEL_FOLDER_PATH = "./savedModel"
DATA_FOLDER_PATH = "./data"

print("Loading model...")
startTime = time.clock()
with open(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_structure.json"), "r") as f:
    loadedModelStructure = f.read()
model = model_from_json(loadedModelStructure)
model.load_weights(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_weights_10_epochs.h5"))
endTime = time.clock()
print("Model is loaded in {0:.2f} seconds".format(endTime - startTime))

df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "fer2013.csv"), header=0)

# Choose random <nb_tests> images and predict them
nb_tests = 16

# [{'image':numpyArray, 'label':'happy', 'accuracy':99.10, 'raw_prediction':[...], 'original_label':2}, {...}, ...]
prediction_data = []

for i in range(nb_tests):
    # String to int array
    rndNumber = random.randint(0, df.shape[0])
    img = df["pixels"][rndNumber].split(" ")
    original_label = df["emotion"][rndNumber]
    img = np.array(img, np.uint8)
    # 1D to 2D
    img = np.reshape(img, (48, 48))

    startTime = time.clock()
    processedImage = preprocessImage(img)
    imgForPrediction = np.reshape(processedImage, (1, 1, 48, 48))

    raw_prediction = model.predict_proba(imgForPrediction, verbose=0)[0]
    prediction_index = np.argmax(raw_prediction)
    prediction_confidence = raw_prediction[prediction_index]
    prediction_label = emotionDict[prediction_index]
    original_label = emotionDict[original_label]
    endTime = time.clock()

    print("Predicted label: {0} with {1:.2f}% confidence in {2:.3f} senconds".format(prediction_label,
                                                                                     (prediction_confidence * 100),
                                                                                     endTime - startTime))

    p_data = {'image': img, 'label': prediction_label, 'confidence': prediction_confidence,
              'raw_prediction': raw_prediction, 'original_label': original_label}
    prediction_data.append(p_data)


# Visualize the predictions
sqrtNumber = findBiggerSqrtNumber(len(prediction_data))
fig = plt.figure(figsize=(10, 10))
for i, p in enumerate(prediction_data):
    img = p['image']
    label = p['label']
    conf = p['confidence']
    raw = p['raw_prediction']
    o_label = p['original_label']

    ax = fig.add_subplot(sqrtNumber, sqrtNumber, i + 1)
    ax.imshow(img, cmap=matplotlib.cm.gray)

    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{0} in {1:.2f}%".format(label, conf))
    plt.tight_layout()

plt.show()
