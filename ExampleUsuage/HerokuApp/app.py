"""
 *****************************************************
 *
 *              Gabor Vecsei
 * Email:       vecseigabor.x@gmail.com
 * Blog:        https://gaborvecsei.wordpress.com/
 * LinkedIn:    www.linkedin.com/in/vecsei-gabor
 * Github:      https://github.com/gaborvecsei
 *
 *****************************************************/
"""

import time
import cv2
import numpy as np
from flask import Flask, jsonify
from ModelLoader import ModelLoader

# Load model in a different thread
modelLoader = ModelLoader("path/to/model_structure.json", "path/to/model_weights.h5")
modelLoader.start()

emotionDict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

app = Flask(__name__)


@app.route('/')
def homepage():
    welcomeLabel = "<h1>Welcome at Emotion Recognition by Gabor Vecsei!</h1>"
    return welcomeLabel


@app.route('/isModelLoaded')
def isModelLoaded():
    loaded = False
    if modelLoader.getModel() is not None:
        loaded = True
    return jsonify(is_model_loaded=loaded)


def preprocessImageForPrediction(image):
    pass


def base64StringToImage(base64Str):
    pass


@app.route('/predict/<base64Str>')
def predict(base64Str):
    pass


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
