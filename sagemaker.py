import numpy as np
import os
import tensorflow.python as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

from data import IesnData

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# Input / Output shape
IMG_SIZE = 512
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = 9

iesn_data = IesnData()


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
                                                weights='imagenet')(features[INPUT_TENSOR_NAME])
    base_model.trainable = False
    pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    output_layer = tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax')(pooling)

    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"emotion": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"emotion": predictions})})

    # 2. Define the loss function for training/evaluation using Tensorflow.
    loss = tf.losses.softmax_cross_entropy(labels, predictions)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")

    # 4. Generate necessary evaluation metrics.
    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=IMG_SHAPE)
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'image-emotion-train.json')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'image-emotion-eval.json')


def _input_fn(training_dir, training_filename):
    metadata_path = os.path.join(training_dir, training_filename)
    return IesnData(metadata_path)\
        .get_train_dataset(tf.keras.applications.resnet50.preprocess_input, BATCH_SIZE, 10)
