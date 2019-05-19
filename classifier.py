import argparse
import os
import pathlib

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

INPUT_TENSOR_NAME = "image"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Input / Output shape
IMG_SIZE = 128
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = 9


def build_classifier():
    # Create the base model from the pre-trained ResNet50 network
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                pooling="avg",
                                                weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    preds = Dense(OUTPUT_SHAPE, activation='softmax')(x)
    return Model(inputs=base_model.inputs, outputs=preds)


def _input_size(directory, glob="*"):
    filepaths = map(lambda x: str(x), pathlib.Path(directory).glob(glob))
    return len(list(filepaths))


def _input_fn(directory,
              glob="*",
              limit=None,
              buffer_size=2 * BATCH_SIZE,
              batch_size=BATCH_SIZE,
              prefetch_batch_num=2):
    filepaths = map(lambda x: str(x), pathlib.Path(directory).glob(glob))
    filepaths = list(filepaths)
    files = tf.data.Dataset.from_tensor_slices(filepaths)
    if limit is not None:
        files = files.take(limit)
    image_preprocessor = tf.keras.applications.resnet50.preprocess_input
    dataset = tf.data.TFRecordDataset(files) \
        .map(lambda x: _parse_tfrecord_(x)) \
        .map(lambda x, y: (image_preprocessor(x), y)) \
        .shuffle(buffer_size=buffer_size) \
        .repeat() \
        .batch(batch_size) \
        .prefetch(buffer_size=prefetch_batch_num)
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    image, label = iterator.get_next()
    # Create a one hot array for your labels
    label = tf.one_hot(label, OUTPUT_SHAPE)
    return image, label


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default="./model")

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    args, _ = parser.parse_known_args()

    # os.environ["AWS_REGION"] = "ap-southeast-2"
    # os.environ["S3_ENDPOINT"] = "s3.ap-southeast-2.amazonaws.com"

    classifier = build_classifier()
    classifier.compile(optimizer=Adam(lr=args.learning_rate),
                       loss=categorical_crossentropy,
                       metrics=['accuracy'])

    output_dir = os.environ.get("SM_MODEL_DIR")
    checkpoint_path = os.path.join(output_dir, "checkpoint_classifier.ckpt")
    cp_local_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=1)

    s3_cp_path = args.model_dir + "/checkpoint_classifier.ckpt"
    cp_s3_callback = tf.keras.callbacks.ModelCheckpoint(s3_cp_path,
                                                        save_weights_only=True,
                                                        verbose=1)
    train_size = _input_size(args.train)
    train_x, train_y = _input_fn(args.train)

    eval_size = _input_size(args.eval)
    eval_x, eval_y = _input_fn(args.eval)

    steps_per_epoch_train = max(train_size // BATCH_SIZE, 1)
    steps_per_epoch_eval = max(eval_size // BATCH_SIZE, 1)
    classifier.fit(train_x, train_y,
                   epochs=args.epochs,
                   steps_per_epoch=steps_per_epoch_train,
                   validation_data=(eval_x, eval_y),
                   validation_steps=steps_per_epoch_eval,
                   callbacks=[cp_local_callback, cp_s3_callback])

    classifier.save(os.path.join(output_dir, "trained_model.h5"))
