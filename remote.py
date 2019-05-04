from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='classifier.py',
                       role="AmazonSageMaker-ExecutionRole-20190430T195525",
                       framework_version='1.12.0',
                       py_version='py3',
                       train_instance_count=1,
                       train_instance_type='ml.m5.large',
                       base_job_name='emotion-label-classifier',
                       hyperparameters={
                           'learning_rate': 0.001,
                           'epochs': 10,
                           'train_path': 's3://konrad-data-storage/image-emotion-100k.json',
                           'eval_path': 's3://konrad-data-storage/image-emotion-subset.json'
                       })

estimator.fit({
    "train": "s3://konrad-data-storage/image-emotion-100k.json",
    "eval": "s3://konrad-data-storage/image-emotion-subset.json"
})
