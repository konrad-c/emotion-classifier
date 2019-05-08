import boto3
import os
import argparse
import json
import pathlib

from tensorflow.python.lib.io import file_io
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput
import tensorflow as tf

INPUT_TENSOR_NAME = "image"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001
BATCH_SIZE = 5

# Input / Output shape
IMG_SIZE = 128
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = 9


def model_fn(features, labels, mode, params):
    """
    Model function for Estimator.
     # Logic to do the following:
     # 1. Configure the model via Keras functional api
     # 2. Define the loss function for training/evaluation using Tensorflow.
     # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
     # 4. Generate predictions as Tensorflow tensors.
     # 6. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
     """

    # 1. Configure the model via Keras functional api
    # Create the base model from the pre-trained ResNet50 network
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                # pooling="avg",
                                                weights='imagenet')(features[INPUT_TENSOR_NAME])
    base_model.trainable = False
    pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    output_layer = tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax')(pooling)

    predictions = tf.reshape(output_layer, [1, OUTPUT_SHAPE])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"emotion": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"emotion": predictions})})

    # 2. Define the loss function for training/evaluation using Tensorflow.
    cross_entropy = tf.losses.softmax_cross_entropy(labels, predictions)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    optimizer = tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])

    train_op = optimizer.minimize(
        loss=cross_entropy,
        global_step=tf.train.get_global_step())

    # 4. Generate necessary evaluation metrics.
    # label_tensor = tf.cast(labels, tf.float32)
    # accuracy, accuracy_update_op = tf.metrics.accuracy(labels=tf.argmax(label_tensor), predictions=tf.argmax(predictions), name='accuracy_op')
    # metrics = {
    #     'accuracy': (accuracy, accuracy_update_op),
    #     'cross_entropy': (cross_entropy, train_op)
    # }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op)
        # eval_metric_ops=metrics)


def serving_input_fn():
    tensor = tf.placeholder(tf.float32, shape=IMG_SHAPE)
    receiver_tensors = { INPUT_TENSOR_NAME: tensor }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def _input_fn(directory,
            limit=None,
            buffer_size=2 * BATCH_SIZE,
            batch_size=BATCH_SIZE,
            prefetch_batch_num=2):
    filepath_generator = pathlib.Path(directory).iterdir()
    files = tf.data.Dataset.from_generator(filepath_generator, tf.string, (tf.TensorShape([None])))
    # files = tf.data.Dataset.list_files(directory, shuffle=True)
    if limit is not None:
        files = files.take(limit)
    image_preprocessor = tf.keras.applications.resnet50.preprocess_input
    return tf.data.TFRecordDataset(files) \
        .map(lambda x: _parse_tfrecord_(x)) \
        .map(lambda x, y: (image_preprocessor(x), y)) \
        .map(lambda x, y: ({"image": x}, tf.one_hot(y, OUTPUT_SHAPE))) \
        .shuffle(buffer_size=buffer_size) \
        .repeat() \
        .batch(batch_size) \
        .prefetch(buffer_size=prefetch_batch_num)


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

    configuration = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=5,
        save_checkpoints_steps=1,
        log_step_count_steps=1)  # set the frequency of logging steps for loss function

    params = {
        "learning_rate": args.learning_rate,
        "total_steps": 100
    }

    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=configuration)

    if args.train:
        estimator.train(input_fn=lambda : _input_fn(args.train), steps=args.epochs)
    elif args.eval:
        scores = estimator.evaluate(input_fn=lambda : _input_fn(args.eval), steps=5)
        print("Evaluation scores: " + scores)
