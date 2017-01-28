import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from customCallbacks import LogTraining
from dataGenerator import dataGenerator

K.set_image_dim_ordering('th')

# Magic...
tf.python.control_flow_ops = tf

emotionDict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

# Train data
# X: (n_samples, 1, rows, cols)
# Y: (n_samples, n_category)
X_train = np.load("data/X_train.npy").astype(np.float16)
# Load and one-hot-encode
y_train = to_categorical(np.load("data/Y_train.npy").astype(np.uint8))

print("X shape: {0}\nY shape: {1}".format(X_train.shape, y_train.shape))

# (rows, cols)
image_shape = (X_train.shape[2], X_train.shape[3])
# (1, rows, cols)
train_image_shape = (1,) + image_shape
print("Image shape: {0}".format(image_shape))
print("Train image shape: {0}".format(train_image_shape))

batch_size = 256
nb_epoch = 50

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=train_image_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotionDict), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkPoint = ModelCheckpoint("savedModel/weights-{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5", monitor="loss",
                             save_best_only=True,
                             save_weights_only=True)
logTraining = LogTraining("savedModel/training_log.txt")
callbacks = [checkPoint, logTraining]

# hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2, shuffle=True, verbose=1, callbacks=callbacks)
startTime = time.clock()
hist = model.fit_generator(dataGenerator(batch_size, X_train, y_train), samples_per_epoch=20480, nb_epoch=nb_epoch, verbose=1,
                           callbacks=callbacks)
endTime = time.clock()

print("Model is trained in {0} seconds!".format(endTime - startTime))

print ("Evaluating the model...")
X_test = np.load("data/X_test.npy").astype(np.float16)
y_test = to_categorical(np.load("data/y_test.npy").astype(np.uint8))
metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print("Metrics: {0}".format(metrics))
print("Model loss: {0}".format(metrics[0]))
print("Model acc: {0}".format(metrics[1]))

print("Saving model...")
modelJson = model.to_json()
with open("savedModel/model_structure.json", "w") as json_file:
    json_file.write(modelJson)
model.save_weights("savedModel/model_weights.h5")
model.save("savedModel/trained_model.h5")
print("Model is saved!")

plt.figure(figsize=(20, 10))
plt.plot(hist.history['loss'])
plt.title("Model loss")
plt.xlabel("epoch")
plt.legend(['loss'], loc='upper left')
plt.savefig("data/train_loss_visualization.png")

plt.figure(figsize=(20, 10))
plt.plot(hist.history['acc'])
plt.title("Model acc")
plt.xlabel("epoch")
plt.legend(['acc'], loc='upper left')
plt.savefig("data/train_acc_visualization.png")
