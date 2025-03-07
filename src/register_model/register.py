from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri('http://ec2-34-201-37-245.compute-1.amazonaws.com:5000/')

client = MlflowClient()

run_id = 'e1ab8d1d93d2498f9b4bdd8fbc2abf15'

model_path = 's3://akshats3dvc/890802656162059317/e1ab8d1d93d2498f9b4bdd8fbc2abf15/artifacts/gradient descent best model'

model_uri = f'runs:/{run_id}/{model_path}'

model_name = 'bestEstimator'

result = mlflow.register_model(model_uri,model_name)

import time
time.sleep(4)

client.update_model_version(
    name = model_name,
    version=result.version,
    description='registering the model with code'
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key='new model',
    value='by code'
)