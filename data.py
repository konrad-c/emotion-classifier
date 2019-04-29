import boto3
import json
from tensorflow.python.lib.io import file_io
import tensorflow as tf


class IesnData:
    dataset_path = "image-emotion-subset.json"
    s3bucket = "konrad-data-storage"
    tfrecordset: tf.data.TFRecordDataset

    def __init__(self, local_metadata_path="./image-emotion-subset.json", record_limit=None):
        self.tfrecordset, self.num_records = self._get_tfrecordset_(local_metadata_path, record_limit)

    def _get_tfrecordset_(self, local_path, limit=None):
        try:
            dataset = open(local_path, 'r')
        except FileNotFoundError:
            s3 = boto3.client("s3")
            s3.download_file(self.s3bucket, self.dataset_path, local_path)
            dataset = open(local_path, 'r')
        tfrecord_filenames = []
        size = 0
        for datapoint in dataset:
            filename = 's3://' + self.s3bucket + '/image-emotion-tfrecords/' + json.loads(datapoint)["id"] + '.tfrecord'
            if file_io.file_exists(filename):
                tfrecord_filenames.append(filename)
                size += 1
            if limit is not None and size >= limit:
                break
        return tf.data.TFRecordDataset(tfrecord_filenames), size

    @staticmethod
    def _parse_tfrecord_(serialized_example):
        features = {
            'valence': tf.FixedLenFeature([], tf.float32),
            'arousal': tf.FixedLenFeature([], tf.float32),
            'dominance': tf.FixedLenFeature([], tf.float32),
            'emotion': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
        example = tf.parse_single_example(serialized_example, features)
        height = tf.cast(example['height'], tf.int64)
        width = tf.cast(example['width'], tf.int64)
        depth = tf.cast(example['depth'], tf.int64)
        image = tf.reshape(
            tf.image.decode_jpeg(example['image_raw']),
            [height, width, depth]
        )
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255.0)
        label = tf.cast(example['emotion'], tf.int64)
        return image, label

    def get_train_dataset(self, image_preprocessor, num_classes, batch_size=32, prefetch_batch_num=10, num_repeats=1):
        return self.tfrecordset \
            .map(lambda x: self._parse_tfrecord_(x)) \
            .map(lambda x, y: (image_preprocessor(x), y)) \
            .map(lambda x, y: ({"image": x}, tf.one_hot(y, num_classes))) \
            .shuffle(buffer_size=100) \
            .repeat(num_repeats) \
            .prefetch(buffer_size=batch_size*prefetch_batch_num) \
            .batch(batch_size)
