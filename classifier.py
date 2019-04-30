from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput
import tensorflow as tf

from data import IesnData

INPUT_TENSOR_NAME = "image"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# Input / Output shape
IMG_SIZE = 512
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
    label_tensor = tf.cast(labels, tf.float32)
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels=tf.argmax(label_tensor), predictions=tf.argmax(predictions), name='accuracy_op')
    metrics = {
        'accuracy': (accuracy, accuracy_update_op),
        'cross_entropy': (cross_entropy, train_op)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops=metrics)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=IMG_SHAPE)
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


def train_input_fn(training_filename):
    return _input_fn(training_filename)


def eval_input_fn(eval_filename):
    return _input_fn(eval_filename)


def _input_fn(filename,
              record_limit=None,
              buffer_size=2 * BATCH_SIZE,
              batch_size=BATCH_SIZE,
              prefetch_batch_num=2):
    return IesnData(dataset_path=filename, local_metadata_path='./' + filename, record_limit=record_limit) \
        .get_dataset(tf.keras.applications.resnet50.preprocess_input,
                     num_classes=OUTPUT_SHAPE,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     prefetch_batch_num=prefetch_batch_num)
