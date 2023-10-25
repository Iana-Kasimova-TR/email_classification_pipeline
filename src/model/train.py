import argparse

import fasttext
import mlflow
import pandas as pd
from fast_text_wrapper import FastTextWrapper
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from utils.load_parameters import read_params


def accuracy_measures(y_test, predictions, avg_method, target_names) -> tuple[float]:
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    print("Classification report")
    print("---------------------", "\n")
    print(classification_report(y_test, predictions, target_names=target_names), "\n")
    print("Confusion Matrix")
    print("---------------------", "\n")
    print(confusion_matrix(y_test, predictions), "\n")

    print("Accuracy Measures")
    print("---------------------", "\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)
    return accuracy, precision, recall, f1score


def get_feat_and_target(df, feature, target) -> tuple:
    """
    Get features and target variables seperately from given dataframe and target
    input: dataframe and target column
    output: two dataframes for x and y
    """
    x = df[feature]
    y = df[target]
    return x, y


def get_target_names(df, target) -> list[str]:
    return df[target].unique()


def train_and_evaluate(config_path):
    config = read_params(config_path)
    raw_data_path = config["external_data_config"]["external_data_csv"]
    train_data_path = config["preprocess_data_config"]["output_txt"]
    test_data_path = config["split_data_config"]["test_data_csv"]
    target = config["split_data_config"]["target"]
    feature = config["split_data_config"]["feature"][0]
    epochs = config["fasttext"]["epoch"]

    raw_df = pd.read_csv(raw_data_path)
    raw_df.dropna(inplace=True)
    target_names = get_target_names(raw_df, target)
    test = pd.read_csv(test_data_path)
    test_x, test_y = get_feat_and_target(test, feature, target)

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model = fasttext.train_supervised(
            input=train_data_path, epoch=epochs, autotuneValidationFile="email.valid"
        )

        true_labels = test_y.tolist()
        texts = test_x.tolist()
        predictions = [
            model.predict(text)[0][0].replace("__label__", "") for text in texts
        ]
        accuracy, precision, recall, f1score = accuracy_measures(
            true_labels, predictions, "weighted", target_names
        )

        mlflow.log_param("epochs", epochs)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        model_path = mlflow_config["registered_model_name"]
        model.save_model(model_path)

        artifacts = {"fasttext_model_path": model_path}
        mlflow_pyfunc_model_path = model_path
        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=FastTextWrapper(),
            extra_pip_requirements=mlflow_config["pip_requirements"],
            artifacts=artifacts,
        )
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_path}"
        mlflow.register_model(model_uri=model_uri, name=model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params/params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
