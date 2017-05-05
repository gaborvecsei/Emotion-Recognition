# Emotion recognition from face

Emotion recognition from faces with Deep Learning CNN network.

## Setup

You will need:

- Python 3 and the following packages:
    - OpenCV 3
    - Keras (with Tensorflow backend)
    - Tensorflow
    - Numpy
    - Seaborn

- Try it out:
    1. prepare data
    2. train model
    3. test it
        1. `predict_emotion.py` script
        2. real time emotion recognition from webcam with `real_time_emotion.py`

## Prepare data

We have to prepare the data for the training

- Download the `fer2013.csv` ([source](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data))
    - Move it to `data` folder and rename it to `fer2013.csv` if it is necessary
        - OR
    - Edit the `config.ini` file
- Run the `prepare_data.py` script
- Now you have the data for the training which you can find in the `data` folder

- *X_train* shape: `(nb_samples, 48, 48)`
- *y_train* shape: `(nb_samples,)`

After this you can start the training!

## Training

Run `train_emotion_recognizer.py`

This will run the training and evaluate the trained model with the test data.

## Predict data (Try it out)

You can easily try the trained model with `predict_emotion.py` script

## Real time prediction

TODO

## About

Gábor Vecsei

- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
- vecseigabor.x@gmail.com