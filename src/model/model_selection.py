import argparse
from pprint import pprint

import fasttext
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from utils.load_parameters import read_params


def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    experiment_id = mlflow_config["experiment_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    logged_model = None

    mlflow.set_tracking_uri(remote_server_uri)

    experiment_name = experiment_id
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment["experiment_id"]
    df = mlflow.search_runs([experiment_id], "", order_by=["metrics.f1_score DESC"])
    best_run_id = df.loc[0, "run_id"]
    client = MlflowClient()
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == best_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Staging"
            )
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    loaded_model = fasttext.load_model(loaded_model.metadata.artifact_path)
    loaded_model.save_model(model_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params/params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
    log_production_model(config_path=parsed_args.config)
