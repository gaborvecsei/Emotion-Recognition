import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

emotionDict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

# Train data
# X: (n_samples, 1, 48, 48)
# Y: (n_samples, n_category)
X = np.load("data/X_train.npy")
# Load and one-hot-encode
Y = to_categorical(np.load("data/Y_train.npy"))

print "X shape: {0}\nY shape: {1}".format(X.shape, Y.shape)

image_shape = (X.shape[2], X.shape[3])
train_image_shape = (1,) + image_shape
print "Image shape: {0}".format(image_shape)
print "Train image shape: {0}".format(train_image_shape)

batch_size = 64
nb_epoch = 100

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=train_image_shape))
model.add(Dropout(0.3))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotionDict), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkPoint = ModelCheckpoint("savedModel/weights-{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss",
                             save_best_only=True,
                             save_weights_only=False)
callbacks = [checkPoint]

hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2, shuffle=True,
                 verbose=1, callbacks=callbacks)

print "Model is trained!"

metrics = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
print "Metrics: {0}".format(metrics)
print "Model loss: {0}".format(metrics[0])
print "Model acc: {0}".format(metrics[1])

print "Saving model..."
modelJson = model.to_json()
with open("savedModel/model_structure.json", "w") as json_file:
    json_file.write(modelJson)
model.save_weights("savedModel/model_weights.h5")
model.save('savedModel/model_trained.h5')
print "Model is saved!"

plt.figure(figsize=(20, 10))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model loss")
plt.xlabel("epoch")
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.savefig("data/train_visualization.jpg")
