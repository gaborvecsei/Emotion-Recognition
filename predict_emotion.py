import os
import random
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

from utils import preprocess_image, emotion_dict, find_bigger_sqrt_number

K.set_image_dim_ordering('th')

# Magic...
tf.python.control_flow_ops = tf

############## Load model #############

SAVE_MODEL_FOLDER_PATH = "./savedModel"
MODEL_STRUCTURE_FILE_NAME = "model_structure.json"
MODEL_WEIGHTS_FILE_NAME = "model_weights_30_epochs.h5"
DATA_FOLDER_PATH = "./data"

print("Loading model...")
try:
    start_time = time.clock()
    with open(os.path.join(SAVE_MODEL_FOLDER_PATH, MODEL_STRUCTURE_FILE_NAME), "r") as f:
        loaded_model_structure = f.read()
    model = model_from_json(loaded_model_structure)
    model.load_weights(os.path.join(SAVE_MODEL_FOLDER_PATH, MODEL_WEIGHTS_FILE_NAME))
    end_time = time.clock()
    print("Model is loaded in {0:.2f} seconds".format(end_time - start_time))
except FileNotFoundError as e:
    print(e)
    print("No saved modeil found in {}".format(SAVE_MODEL_FOLDER_PATH))
    print("You should train one first!")
    sys.exit(1)

############# Predict ###############

df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "fer2013.csv"), header=0)

# Number of test images
nb_tests = 16

# Store predictions
# [{'image':numpyArray, 'label':'happy', 'accuracy':99.10, 'raw_prediction':[...], 'original_label':2}, {...}, ...]
prediction_data = []

for i in range(nb_tests):
    # String to int array
    rnd_number = random.randint(0, df.shape[0])
    img = df["pixels"][rnd_number].split(" ")
    original_label_index = df["emotion"][rnd_number]

    img = np.array(img, np.uint8)
    img = np.reshape(img, (48, 48))

    start_time = time.clock()
    processed_image = preprocess_image(img)
    img_for_prediction = np.reshape(processed_image, (1, 1, 48, 48))

    raw_prediction = model.predict_proba(img_for_prediction, verbose=0)[0]
    prediction_index = np.argmax(raw_prediction)
    prediction_confidence = raw_prediction[prediction_index]
    prediction_label = emotion_dict[prediction_index]
    original_label = emotion_dict[original_label_index]
    end_time = time.clock()

    print("Predicted label: {0} with {1:.2f}% confidence in {2:.3f} senconds".format(prediction_label,
                                                                                     (prediction_confidence * 100),
                                                                                     end_time - start_time))

    p_data = {'image': img, 'label': prediction_label, 'confidence': prediction_confidence,
              'raw_prediction': raw_prediction, 'original_label': original_label}

    prediction_data.append(p_data)

############# Visualize the predictions ###########

# find closes sqrt number to the nb_test so we can create a grid for the visualization
sqrt_number = find_bigger_sqrt_number(len(prediction_data))

fig = plt.figure(figsize=(10, 10))

for i, p in enumerate(prediction_data):
    img = p['image']
    label = p['label']
    conf = p['confidence']

    ax = fig.add_subplot(sqrt_number, sqrt_number, i + 1)
    ax.imshow(img, cmap=matplotlib.cm.gray)

    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{0} in {1:.2f}%".format(label, conf))
    plt.tight_layout()

plt.show()
