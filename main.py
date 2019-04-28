import tensorflow.python as tf
import boto3
import json
import os

from tensorflow.python.lib.io import file_io

AUTOTUNE = tf.data.experimental.AUTOTUNE

s3bucket = "konrad-data-storage"
dataset_path = "image-emotion-subset.json"
os.environ["AWS_REGION"] = "ap-southeast-2"
os.environ["S3_ENDPOINT"] = "s3.ap-southeast-2.amazonaws.com"
os.environ["AWS_LOG_LEVEL"] = "3"

IMG_SIZE = 512
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = 9


def get_tfrecord_dataset(limit=None):
    try:
        dataset = open(dataset_path, 'r')
    except FileNotFoundError:
        s3 = boto3.client("s3")
        s3.download_file(s3bucket, dataset_path, dataset_path)
        dataset = open(dataset_path, 'r')
    tfrecord_filenames = []
    size = 0
    for datapoint in dataset:
        filename = 's3://' + s3bucket + '/image-emotion-tfrecords/' + json.loads(datapoint)["id"] + '.tfrecord'
        if file_io.file_exists(filename):
            tfrecord_filenames.append(filename)
            size += 1
        if limit is not None and size >= limit:
            break
    return tf.data.TFRecordDataset(tfrecord_filenames), size


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
    label = tf.cast(example['emotion'], tf.int64)
    return image, label


def tfrecord_train_input_fn(tfdataset: tf.data.TFRecordDataset, record_parser, image_preprocessor, batch_size=32):
    return tfdataset \
        .map(lambda x: record_parser(x)) \
        .map(lambda x, y: (image_preprocessor(x), y)) \
        .shuffle(True) \
        .repeat() \
        .prefetch(buffer_size=batch_size) \
        .batch(batch_size)


def get_model(input_shape, num_outputs):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False

    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_outputs)
    ])


BATCH_SIZE = 32

tfdataset, dataset_size = get_tfrecord_dataset(limit=100)
tfdataset_iterator = tfrecord_train_input_fn(tfdataset,
                                             _parse_tfrecord_,
                                             tf.keras.applications.resnet50.preprocess_input,
                                             batch_size=BATCH_SIZE)
for image, label in tfdataset_iterator.take(5):
    print(label)

model = get_model(IMG_SHAPE, OUTPUT_SHAPE)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

initial_epochs = 10
steps_per_epoch = round(100) // BATCH_SIZE
validation_steps = 1

loss0, accuracy0 = model.evaluate(tfdataset_iterator, steps=validation_steps)
history = model.fit(tfdataset_iterator,
                    epochs=initial_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=tfdataset_iterator,
                    validation_steps=validation_steps)

print(history)

"""
filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png'])  # list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)  # use decode_png or decode_jpeg decoder based on your files.

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(1):  # length of your filename list
    image = my_img.eval()  # here is your image Tensor :)

print(image.shape)
Image.show(Image.fromarray(np.asarray(image)))

coord.request_stop()
coord.join(threads)


# Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        dense_keys=['image_raw', 'label'],
        # Defaults are not specified since both keys are required.
        dense_types=[tf.string, tf.int64])

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image = tf.reshape(image, [my_cifar.n_input])
    image.set_shape([my_cifar.n_input])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


# images and labels array as input
def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
"""
