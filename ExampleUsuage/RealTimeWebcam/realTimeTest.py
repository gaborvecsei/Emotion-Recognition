import os
import time

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

from utils import preprocess_image, emotion_dict

K.set_image_dim_ordering('th')
# Magic...
tf.python.control_flow_ops = tf

SAVE_MODEL_FOLDER_PATH = os.path.abspath('../../savedModel')

print("Loading model...")
start_time = time.clock()
with open(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_structure.json"), "r") as f:
    loadedModelStructure = f.read()
model = model_from_json(loadedModelStructure)
model.load_weights(os.path.join(SAVE_MODEL_FOLDER_PATH, "model_weights_10_epochs.h5"))
end_time = time.clock()
print("Model is loaded in {0:.2f} seconds".format(end_time - start_time))

face_cascade = cv2.CascadeClassifier("face_cascade.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the face out of the video
            roi = gray[y:y + h, x:x + w]

            roi = cv2.resize(roi, (48, 48))
            roi = preprocess_image(roi)
            roi = np.reshape(roi, (1, 1, 48, 48))

            raw_prediction = model.predict_proba(roi, verbose=0)[0]
            label = emotion_dict[np.argmax(raw_prediction)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))

        cv2.imshow("Press q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
