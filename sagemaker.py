from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='classifier.py',
                       framework_version='1.12.0',
                       training_steps=100,
                       evaluation_steps=20,
                       hyperparameters={'learning_rate': 0.001},
                       train_instance_count=1,
                       train_instance_type='ml.c4.xlarge',
                       base_job_name='emotion-label-classifier')

estimator.fit("image-emotion-100k.json")