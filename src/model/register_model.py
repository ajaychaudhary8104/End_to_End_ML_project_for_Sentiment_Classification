# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking

# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Sentiment_Classification.mlflow')
dagshub.init(repo_owner='ajaychaudhary8104', repo_name='End_to_End_ML_project_for_Sentiment_Classification', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Registering model from URI: {model_uri}")
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model_info['run_id'])
        logging.info(f"Run Artifact URI: {run.info.artifact_uri}")

        # Verify artifacts exist
        artifacts = client.list_artifacts(model_info['run_id'], model_info['model_path'])
        if not artifacts:
            logging.error(f"No artifacts found for run_id: {model_info['run_id']} at path: {model_info['model_path']}")
            root_artifacts = client.list_artifacts(model_info['run_id'])
            logging.info(f"Available artifacts at root: {[a.path for a in root_artifacts]}")
            logging.warning("Artifacts not found. Proceeding with registration, but model version may be invalid.")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
