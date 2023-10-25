import argparse

import pandas as pd

from utils.load_parameters import read_params


# TODO: and NLP preprocessing - lemmatization and so on
def preprocess_train_data(config_path):
    """
    convert data into format for fasttext model
    input: pandas dataframe
    output: saved txt file with preprocessed data
    """
    config = read_params(config_path)
    df_path = config["preprocess_data_config"]["path_to_csv"]
    df = pd.read_csv(df_path)
    output_txt = config["preprocess_data_config"]["output_txt"]
    with open(output_txt, "w") as f:
        for _, row in df.iterrows():
            label = row["product"]
            text = row["narrative"]
            f.write(f"__label__{label} {text}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params/params.yaml")
    parsed_args = args.parse_args()
    preprocess_train_data(config_path=parsed_args.config)
    preprocess_train_data(config_path=parsed_args.config)
