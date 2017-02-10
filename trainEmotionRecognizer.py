import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from customCallbacks import LogTraining, SlackNotifier
from dataGenerator import data_generator

K.set_image_dim_ordering('th')
# Magic...
tf.python.control_flow_ops = tf

DATA_FOLDER_PATH = "./data"
TRAIN_X_PATH = os.path.join(DATA_FOLDER_PATH, "X_train.npy")
TRAIN_Y_PATH = os.path.join(DATA_FOLDER_PATH, "Y_train.npy")
TEST_X_PATH = os.path.join(DATA_FOLDER_PATH, "X_test.npy")
TEST_Y_PATH = os.path.join(DATA_FOLDER_PATH, "Y_test.npy")

SAVE_MODEL_FOLDER_PATH = "./savedModel"
CHECKPOINT_FOLDER_PATH = os.path.join(SAVE_MODEL_FOLDER_PATH, "trainCheckpoints")
VISUALIZATION_FOLDER_PATH = os.path.join(SAVE_MODEL_FOLDER_PATH, "visualization")

# Train data
# X: (n_samples, 1, rows, cols)
# Y: (n_samples, n_category)
X_train = np.load(TRAIN_X_PATH).astype(np.uint8)
# Load and one-hot-encode
y_train = to_categorical(np.load(TRAIN_Y_PATH).astype(np.uint8))

print("X shape: {0}\nY shape: {1}".format(X_train.shape, y_train.shape))

# original shape: (rows, cols)
image_shape = (X_train.shape[1], X_train.shape[2])
# train data shape: (1, rows, cols)
train_image_shape = (1,) + image_shape
print("Image shape: {0}".format(image_shape))
print("Train image shape: {0}".format(train_image_shape))
print("Output layer dim: {0}".format(y_train.shape[1]))

batch_size = 64
samples_per_epoch = 20480
nb_epoch = 10

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu', input_shape=train_image_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# existing_model_weights_path = os.path.join(SAVE_MODEL_FOLDER_PATH, "model_weights_10_epochs.h5")
# if os.path.exists(existing_model_weights_path):
#     print("Loading weights...")
#     model.load_weights(existing_model_weights_path)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkPoint = ModelCheckpoint(os.path.join(CHECKPOINT_FOLDER_PATH, "weights-{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5"),
                             monitor="loss",
                             save_best_only=True,
                             save_weights_only=True)
logTraining = LogTraining(os.path.join(VISUALIZATION_FOLDER_PATH, "training_log_{0}_epochs.txt".format(nb_epoch)))
tensorboard = TensorBoard(log_dir=os.path.join(SAVE_MODEL_FOLDER_PATH, "tensorboard_logs"), histogram_freq=0,
                          write_graph=False, write_images=True)
slackNotifier = SlackNotifier("xoxp-133638627843-133707362674-133898083956-36217820cda1e6d43161f0894437bd5a",
                              channelName="notifications")
callbacks = [checkPoint, logTraining, tensorboard, slackNotifier]

startTime = time.clock()
hist = model.fit_generator(data_generator(batch_size, X_train, y_train, augmentation=True),
                           samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                           verbose=1,
                           callbacks=callbacks)
endTime = time.clock()

print("Model is trained in {0} seconds!".format(endTime - startTime))

# Saving the trained model

print("Saving model...")
modelJson = model.to_json()
with open(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_structure.json"), "w") as json_file:
    json_file.write(modelJson)
model.save_weights(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_weights_{0}_epochs.h5".format(nb_epoch)))
model.save(os.path.join(SAVE_MODEL_FOLDER_PATH, "trained_model_{0}_epochs.h5".format(nb_epoch)))
print("Model is saved!")

# Model evaluation

print("Evaluating the model...")
X_test = np.load(TEST_X_PATH).astype(np.float16)
# Prepare data for evaluation
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
y_test = to_categorical(np.load(TEST_Y_PATH).astype(np.uint8))
metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print("Metrics: {0}".format(metrics))
print("Model loss: {0}".format(metrics[0]))
print("Model acc: {0}".format(metrics[1]))

# Visualize the training

plt.figure(figsize=(15, 10))
plt.plot(hist.history['loss'])
plt.title("Model loss")
plt.xlabel("epoch")
plt.legend(['loss'], loc='upper left')
plt.savefig(os.path.join(VISUALIZATION_FOLDER_PATH, "train_loss_visualization_{0}_epochs.png".format(nb_epoch)))

plt.figure(figsize=(15, 10))
plt.plot(hist.history['acc'])
plt.title("Model acc")
plt.xlabel("epoch")
plt.legend(['acc'], loc='upper left')
plt.savefig(os.path.join(VISUALIZATION_FOLDER_PATH, "train_acc_visualization_{0}_epochs.png".format(nb_epoch)))
