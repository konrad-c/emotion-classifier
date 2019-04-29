import os
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput
import tensorflow as tf

from data import IesnData

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

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
                                                pooling="avg",
                                                weights='imagenet')(features[INPUT_TENSOR_NAME])
    base_model.trainable = False
    output_layer = tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax')(base_model)

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
    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        # "rmse": rmse
    }

    train_tensors_log = {
        # 'rmse': rmse,
        'cross_entropy': cross_entropy,
        'global_step': tf.train.get_global_step()
    }
    train_hook_list = [tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=5)]

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op,
        training_hooks=train_hook_list,
        eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=IMG_SHAPE)
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


def train_input_fn(training_dir="./"):
    return _input_fn(training_dir, 'image-emotion-train.json')


def eval_input_fn(training_dir="./"):
    return _input_fn(training_dir, 'image-emotion-eval.json')


def _input_fn(training_dir, training_filename):
    metadata_path = os.path.join(training_dir, training_filename)
    return IesnData(metadata_path, record_limit=30) \
        .get_train_dataset(tf.keras.applications.resnet50.preprocess_input,
                           num_classes=OUTPUT_SHAPE,
                           batch_size=5,
                           prefetch_batch_num=10)


if __name__ == "__main__":
    os.environ["AWS_REGION"] = "ap-southeast-2"
    os.environ["S3_ENDPOINT"] = "s3.ap-southeast-2.amazonaws.com"
    os.environ["AWS_LOG_LEVEL"] = "3"

    params = {"learning_rate": LEARNING_RATE}
    model_dir = "./output"

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=100)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
