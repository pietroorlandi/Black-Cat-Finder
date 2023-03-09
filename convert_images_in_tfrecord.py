import tensorflow as tf
import os
from tqdm import tqdm

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_from_single_images(name_tfrecord="dataset_train.tfrecord"):
    with tf.io.TFRecordWriter(name_tfrecord) as record_writer:
        for label_str in os.listdir(train_dir):
            label_dir = os.path.join(train_dir, label_str)
            for image_path in os.listdir(label_dir):
                image_abs_path = os.path.join(label_dir, image_path)
                image = tf.io.read_file(image_abs_path)
                label = int(label_str)
                tfrecord_example = tf.train.Example(
                       features=tf.train.Features(
                            feature={
                                  "image": _bytes_feature(image),
                                  "label": _int64_feature(label)
                             }
                       )
                    )
                record_writer.write(tfrecord_example.SerializeToString())
        print(f"TFRecord created correctly: new file is {name_tfrecord}")

data_dir = r"Data"
train_dir = os.path.join(data_dir,"Train")
test_dir = os.path.join(data_dir,"Test")
create_tfrecord_from_single_images()
