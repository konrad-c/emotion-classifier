import tensorflow as tf
import os

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Shapes of training, test and validation set
print("Fashion MNIST:")
print("Training set (images) shape: {shape}".format(shape=train_images.shape))
print("Training set (labels) shape: {shape}".format(shape=train_labels.shape))

print("Test set (images) shape: {shape}".format(shape=test_images.shape))
print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_mnist_fashion_dataset(images, labels, name, directory):
    _, height, width = images.shape

    filename = os.path.join(directory, name + '.tfrecords')
    print(f'Writing {filename}')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(len(images)):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(1),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())


convert_mnist_fashion_dataset(train_images, train_labels, 'train', 'data')
convert_mnist_fashion_dataset(test_images, test_labels, 'validation', 'data')

import sagemaker

bucket = sagemaker.Session().default_bucket()  # Automatically create a bucket
prefix = 'radix/mnist_fashion_tutorial'  # Subfolder prefix

s3_url = sagemaker.Session().upload_data(path='data',
                                         bucket=bucket,
                                         key_prefix=prefix + '/data/mnist')
print(s3_url)
