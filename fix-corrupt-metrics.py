#!/usr/bin/env python
"""Fix metrics by reparsing corrupt log files

"""
import ast
from pathlib import Path

import mlflow

from scanprops import evt_rate_duration_from_log
from selections import read_config_toml, select_runs


def read_corrupt_logs(file_path: str | Path) -> str:
    """Read corrupt log files and try to decode as text"""

    log_bin = Path(file_path).read_bytes()
    return log_bin.decode("unicode_escape").strip(" \n'\"")


def print_corrupt_logs(file_path: str | Path):
    """Print corrupt log files as text"""
    print(read_corrupt_logs(file_path))


def parse_metrics(df):
    """Parse metrics from corrupt log files

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `params.use_int8`, `run_id`, and `artifact_uri`

    """
    use_int8 = df["params.use_int8"].map(ast.literal_eval).fillna(False)
    df = df[use_int8].copy()
    df["artifact_uri"] = df["artifact_uri"].str.replace(
        "file:///home/sali/codebaby/run-allen-run/", ""
    )
    print("candidates: ", len(df))
    for _, row in df.iterrows():
        with mlflow.start_run(run_id=row["run_id"]):
            log_file, *_ = Path(row["artifact_uri"]).glob("stdout-*")
            try:
                log_txt = read_corrupt_logs(log_file)
                evt_rate, duration = evt_rate_duration_from_log(log_txt)
            except ValueError:
                print(f"Failed to parse log file: {log_file}")
                # print_corrupt_logs(log_file)
                continue
            mlflow.log_metric("evt_rate", evt_rate)
            mlflow.log_metric("duration", duration)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toml_config", help="TOML file with run selection")
    opts = parser.parse_args()

    selection = read_config_toml(opts.toml_config)["runs"]["ghostbuster"][0].values()
    runs = select_runs([selection])
    parse_metrics(runs)
