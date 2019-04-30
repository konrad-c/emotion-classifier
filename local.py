import os
import classifier
import tensorflow as tf

dataset_path = "image-emotion-subset.json"
s3bucket = "konrad-data-storage"
os.environ["AWS_REGION"] = "ap-southeast-2"
os.environ["S3_ENDPOINT"] = "s3.ap-southeast-2.amazonaws.com"
os.environ["AWS_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.INFO)


def get_estimator():
    configuration = tf.estimator.RunConfig(
        model_dir="./output",
        keep_checkpoint_max=5,
        save_checkpoints_steps=5,
        log_step_count_steps=1)  # set the frequency of logging steps for loss function

    params = {
        "learning_rate": 0.001,
        "total_steps": 100
    }

    return tf.estimator.Estimator(model_fn=classifier.model_fn, params=params, config=configuration)


# Local arguments
cur_mode = tf.estimator.ModeKeys.TRAIN

filename = "s3://konrad-data-storage/image-emotion-subset.json"
estimator = get_estimator()

if cur_mode == tf.estimator.ModeKeys.TRAIN:
    local_train_input = lambda: classifier._input_fn(filename, record_limit=30, buffer_size=5,
                                                     batch_size=5, prefetch_batch_num=1)
    estimator.train(local_train_input, steps=20)
elif cur_mode == tf.estimator.ModeKeys.EVAL:
    local_eval_input = lambda: classifier._input_fn(filename, record_limit=30, buffer_size=5,
                                                    batch_size=5, prefetch_batch_num=1)
    estimator.evaluate(local_eval_input, steps=5)
