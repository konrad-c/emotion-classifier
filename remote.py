from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='classifier.py',
                       role="AmazonSageMaker-ExecutionRole-20190430T195525",
                       framework_version='1.12.0',
                    #    model_dir='s3://konrad-data-storage/emotion-classifier-labels',
                       py_version='py3',
                       train_instance_count=1,
                     #   train_instance_type='local',
                       train_instance_type='ml.p2.xlarge',
                       base_job_name='emotion-label-classifier',
                       hyperparameters={
                           'learning_rate': 0.001,
                           'epochs': 10,
                        #    'eval_path': 's3://konrad-data-storage/image-emotion-tfrecords-128/0003438f-9abf-4bd2-9637-8a79c2df248d.tfrecord'
                       })


estimator.fit({
    "train": "s3://konrad-data-storage/image-emotion-tfrecords-128-subset",
    "eval": "s3://konrad-data-storage/image-emotion-tfrecords-128-subset"
})