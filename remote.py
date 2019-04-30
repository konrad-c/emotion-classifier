from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='classifier.py',
                       role="AmazonSageMaker-ExecutionRole-20190430T195525",
                       framework_version='1.12.0',
                       training_steps=10,
                       evaluation_steps=5,
                       hyperparameters={'learning_rate': 0.001},
                       train_instance_count=1,
                       train_instance_type='ml.c4.xlarge',
                       base_job_name='emotion-label-classifier')

estimator.fit("s3://konrad-data-storage/image-emotion-100k.json")