import argparse

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.load_parameters import read_params


def split_stratified_data(
    df, target, feature, test_data_path, train_data_path, split_ratio, random_state
):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=split_ratio, random_state=random_state
    )
    X = df[feature]
    y = df[target]
    for train_idx, test_idx in splitter.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(train_data_path)
    test.to_csv(test_data_path)


def split_data(config_path):
    """
    split the train dataset(data)
    input: config path
    output: return splitted dataset
    """
    config = read_params(config_path)
    raw_data_path = config["external_data_config"]["external_data_csv"]
    split_ratio = config["split_data_config"]["train_test_split_ratio"]
    random_state = config["split_data_config"]["random_state"]
    target = config["split_data_config"]["target"]
    feature = config["split_data_config"]["feature"]
    test_data_path = config["split_data_config"]["test_data_csv"]
    train_data_path = config["split_data_config"]["train_data_csv"]
    raw_df = pd.read_csv(raw_data_path)
    raw_df.dropna(inplace=True)
    split_stratified_data(
        raw_df,
        target,
        feature,
        test_data_path,
        train_data_path,
        split_ratio,
        random_state,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params/params.yaml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)
    split_data(config_path=parsed_args.config)
