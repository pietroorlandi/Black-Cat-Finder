import tensorflow as tf
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import nvidia.dali.plugin.tf as dali_tf
import tensorflow.compat.v1 as tf_v1
import time
from datetime import datetime
from nvidia import dali
from keras.utils.layer_utils import count_params
import pandas as pd
import numpy as np
import nvidia.dali.tfrecord as tfrec
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping


SIZE_DATASET = 3802
BATCH_SIZE = 32
EPOCHS = 75
tfrecord_train = r"dataset_train.tfrecord"
tfrecord_idx_train = r"idx_files/dataset_train.idx"


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


# Set DALI pipeline for training
@pipeline_def
def pipeline_training():
    random_hue = fn.random.uniform(range=[-10,+35.]) 
    random_saturation = fn.random.uniform(range=[0.6,1.4]) # [-0.4, +0.4]
    random_brightness = fn.random.uniform(range=[0.8,1.3]) # [-0.2, +0.3]
    random_contrast = fn.random.uniform(range=[0.9,1.1]) # [-0.1, +0.1]
    random_angle = fn.random.uniform(range=[0,15]) # angle
    random_shear = fn.random.uniform(range=[-0.1745,0.1745]) # [-10, +10] obtained by [-10, +10] * np.pi / 180
    random_resize = fn.random.uniform(range=[0.75,1.25]) 
    random_horizontal_flip = fn.random.uniform(values=[1,-1]) # -1 is flip , +1 not flip
    random_vertical_flip = fn.random.uniform(values=[1,1]) # -1 is flip , +1 not flip 
    # Make also data-augmentation with probability p=0.5
    inputs = fn.readers.tfrecord(   # Read training-data from disk
        path=tfrecord_train,
        index_path=tfrecord_idx_train,
        random_shuffle=True,
        initial_fill=1800,
        features={   #  (see https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file_2 for more information ) 
            "image" : tfrec.FixedLenFeature((), tfrec.string, ""),  
            "label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
        })
    labels = inputs['label'] # int64
    images = inputs["image"] # flattened uint8 array in a range (0, 255)
    images = fn.decoders.image(images)
    images = fn.reshape(images, [96, 96, 3]) 
    images = images.gpu()
    labels = labels.gpu()
    augmented = fn.copy(images) 
    mt = fn.transforms.rotation(angle=random_angle, center=(48,48))
    mt = fn.transforms.scale(mt, scale=fn.stack(random_horizontal_flip, random_vertical_flip), center=(48.,48.)) # flip random (horizontal)
    mt = fn.transforms.scale(mt, scale=fn.stack(random_resize, random_resize), center=(48.,48.)) # random-scaling
    mt = fn.transforms.shear(mt, shear=fn.stack(random_shear, random_shear), center=(48,48))
    augmented  = fn.warp_affine(augmented, matrix = mt)
    augmented = fn.hsv(augmented, hue=random_hue, saturation=random_saturation, value=random_brightness) 
    augmented  = fn.contrast(augmented, contrast=random_contrast)
    bool_is_augmented = fn.random.coin_flip(probability=0.5, dtype=types.DALIDataType.BOOL)
    bool_not_augmented = bool_is_augmented ^ True
    out_images = bool_is_augmented * augmented + bool_not_augmented * images
    return out_images, labels

@pipeline_def
def pipeline_validation():
    jpegs, labels = fn.readers.file(file_root="Data/Test/", random_shuffle=False)
    labels = fn.cast(labels, dtype=types.INT64)
    images = fn.decoders.image(jpegs)
    images = fn.reshape(images, [96, 96, 3])
    images = images.gpu() # Copy the data on GPU
    labels = labels.gpu()
    return images, labels

def main():
    shapes = ((BATCH_SIZE, 96, 96, 3), (BATCH_SIZE, 1))
    with tf.device('/gpu:0'):
        pipe_train = pipeline_training(batch_size=BATCH_SIZE, num_threads=4, device_id=0)
        pipe_validation = pipeline_validation(batch_size=BATCH_SIZE, num_threads=4, device_id=0)
        # Create dataset
        dataset_train = dali_tf.DALIDataset(   # for more information see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/tensorflow_plugin_api.html
            pipeline=pipe_train,
            batch_size=BATCH_SIZE,
            output_shapes=shapes,
            output_dtypes=(tf.uint8, tf.int64),
            device_id=0)
        dataset_validation = dali_tf.DALIDataset(   # for more information see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/tensorflow_plugin_api.html
            pipeline=pipe_validation,
            batch_size=BATCH_SIZE,
            output_shapes=shapes,
            output_dtypes=(tf.uint8, tf.int64),
            device_id=0)

        # Definition of CNN
        model = tf.keras.models.Sequential(name="Cat-Classification-CNN")
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

        # Callback declariations
        early_stopping = EarlyStopping(patience=25, monitor="val_loss", verbose=1)
        filepath_model = r"cnn_cat_model.h5"
        checkpoint = ModelCheckpoint(filepath_model, monitor='val_loss',mode='min',save_best_only=True,verbose=1)


        history = model.fit(
                dataset_train,
                validation_data = dataset_validation,
                validation_steps=1,
                steps_per_epoch=3802//BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[checkpoint, early_stopping]
                )
        return model, history
    

if __name__=='__main__':
    model, history = main()

