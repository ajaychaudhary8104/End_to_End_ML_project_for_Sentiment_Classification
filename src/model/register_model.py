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
        #model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        #model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
    
        # 1. Get the latest version number
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    
        # 2. Transition it
        client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=True
    )
        print(f"✅ Version {latest_version} of '{model_name}' moved to Stagging.")

        
        
        logging.debug('Model %s registered successfully with info: %s', model_name, model_info)
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "Sentiment_Classification_Model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

