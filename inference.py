import tensorflow as tf
import os
import tensorflow.compat.v1 as tf_v1
import time
from datetime import datetime
import numpy as np
from tensorflow import keras
from keras import backend as K

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
    filepath_model = r"cnn_cat_model.h5"
    model = tf.keras.models.load_model(filepath_model)
    path = "Data/Test/0/address1_picture27.jpg"
    test_sample = tf.io.read_file(path)
    test_sample = tf.io.decode_jpeg(test_sample)
    test_sample = tf.reshape(test_sample, [1, 96, 96, 3])
    #test_samples= tf.random.uniform([4, 96, 96, 3], minval=0, maxval=1)
    #test_samples = test_samples * 255
    #test_samples= tf.cast(test_samples, dtype=tf.uint8)
    y_pred_batch = model.predict(test_sample)
    print(f"Prediction: {y_pred_batch}")
    

if __name__=='__main__':
    main()

