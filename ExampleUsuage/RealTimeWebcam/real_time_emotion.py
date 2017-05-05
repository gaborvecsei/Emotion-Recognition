import os
import time

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

from utils import preprocess_image, EMOTION_DICT

K.set_image_dim_ordering('th')
tf.python.control_flow_ops = tf

SAVED_MODEL_FOLDER_PATH = os.path.abspath('../../created_models/30_epoch_training')
SAVED_MODEL_STRUCTURE_FILE_PATH = os.path.join(SAVED_MODEL_FOLDER_PATH, "model_structure.json")
SAVED_MODEL_WEIGHTS_FILE_PATH = os.path.join(SAVED_MODEL_FOLDER_PATH, "model_weights_30_epochs.h5")
FACE_CASCADE_FILE_PATH = "./face_cascade.xml"

print("Loading model...")
start_time = time.clock()
with open(SAVED_MODEL_STRUCTURE_FILE_PATH, "r") as f:
    loaded_model_structure = f.read()
model = model_from_json(loaded_model_structure)
model.load_weights(SAVED_MODEL_WEIGHTS_FILE_PATH)
end_time = time.clock()
print("Model is loaded in {0:.2f} seconds".format(end_time - start_time))

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE_PATH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = preprocess_image(face_roi)
            face_roi = np.reshape(face_roi, (1, 1, 48, 48))

            raw_prediction = model.predict_proba(face_roi, verbose=0)[0]
            label = EMOTION_DICT[np.argmax(raw_prediction)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))

        cv2.imshow("Press q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
