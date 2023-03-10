import tensorflow as tf
import os
import tensorflow.compat.v1 as tf_v1
import time
from datetime import datetime
import numpy as np
from tensorflow import keras
from keras import backend as K
import cv2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main():

# load the saved model

# create a VideoCapture object for the webcam
    cap = cv2.VideoCapture('/dev/video2') # external webcam
    filepath_model = r"cnn_cat_model.h5"
    model = tf.keras.models.load_model(filepath_model)

    while True:
        # read a frame from the webcam
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize the frame to match the input shape of the model
        resized_frame = cv2.resize(frame, (96, 96))
        

        # convert the resized frame to a numpy array and normalize its values
        x = np.array(resized_frame)

        # add a batch dimension to the input array
        x = np.expand_dims(x, axis=0)

        # make a prediction using the loaded model
        y_pred = model.predict(x)
        print(y_pred)
        # get the predicted label from the prediction array
        label = "False" if y_pred<0.5 else "Black cat"

        # draw the predicted label on the frame
        cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # show the frame on the screen
        cv2.imshow('frame', frame)

        # break the loop if the 'q' key is pressed
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    

if __name__=='__main__':
    main()

