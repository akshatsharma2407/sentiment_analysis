from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri('http://ec2-34-201-37-245.compute-1.amazonaws.com:5000/')

client = MlflowClient()

model_name = 'bestEstimator'

model_version = 2

new_stage = 'Production'

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)
