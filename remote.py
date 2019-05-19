from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='classifier.py',
                       role="AmazonSageMaker-ExecutionRole-20190430T195525",
                       framework_version='1.12.0',
                       model_dir='s3://konrad-data-storage/emotion-classifier-labels',
                       py_version='py3',
                       train_instance_count=1,
                     #   train_instance_type='local',
                     #   train_instance_type='ml.m5.large',
                       train_instance_type='ml.p2.xlarge',
                       base_job_name='emotion-label-classifier',
                       hyperparameters={
                           'learning_rate': 0.001,
                           'epochs': 10,
                       })


estimator.fit({
    "train": "s3://konrad-data-storage/image-emotion-tfrecords-128",
    "eval": "s3://konrad-data-storage/image-emotion-tfrecords-128-eval"
})