import argparse
import os

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.orm import sessionmaker

from load_parameters import read_params

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATABASE_URL = "sqlite:///" + os.path.join(basedir, "email.db")


def model_monitoring(config_path):
    config = read_params(config_path)
    train_data_path = config["external_data_config"]["external_data_csv"]
    monitor_dashboard_path = config["model_monitoring"]["monitor_dashboard_path"]

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    metadata = MetaData()
    metadata.reflect(engine)
    my_table = Table("sample", metadata, autoload_with=engine)
    results = session.query(my_table).all()

    prod = pd.DataFrame(
        [(d.input, d.result) for d in results], columns=["narrative", "product"]
    )

    ref = pd.read_csv(train_data_path)

    prod["product"] = pd.factorize(prod["product"])[0]
    ref["product"] = pd.factorize(ref["product"])[0]

    prod = prod.reset_index(drop=True)
    ref = ref.reset_index(drop=True)

    column_mapping = ColumnMapping(target="product", text_features=["narrative"])
    print(prod)
    data_drift_report = Report(
        metrics=[
            DataDriftPreset(
                num_stattest="ks",
                cat_stattest="psi",
                num_stattest_threshold=0.2,
                cat_stattest_threshold=0.2,
            ),
        ]
    )
    data_drift_report.run(
        reference_data=ref[["narrative", "product"]],
        current_data=prod[["narrative", "product"]],
        column_mapping=column_mapping,
    )
    data_drift_report.save_html(monitor_dashboard_path)
    session.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)
