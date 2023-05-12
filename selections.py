#!/usr/bin/env python

from datetime import datetime
from typing import cast

import tomlkit

import mlflow
import pandas as pd


def read_config_toml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return tomlkit.parse(f.read())


def unix_ts(time: str) -> int:
    """Unix timestamp in milliseconds."""
    return int(datetime.strptime(time, "%Y-%m-%d %H:%M").timestamp()) * 1000


def select_runs(selections: list[tuple[str, str, str]]) -> pd.DataFrame:
    """
    Select runs from MLflow server.
    """
    query_fmt = [
        "status = 'FINISHED'",
        "start_time > {start}",
        "start_time < {stop}",
        "params.sequence LIKE '{seq}%'",
    ]
    runs = [
        cast(
            pd.DataFrame,
            mlflow.search_runs(
                experiment_names=["v100"],
                filter_string=" AND ".join(query_fmt).format(
                    start=unix_ts(start), stop=unix_ts(stop), seq=seq
                ),
            ),
        )
        for start, stop, seq in selections
    ]
    # FIXME: workaround, LIKE isn't working
    runs = [
        df[df["params.sequence"].str.startswith(seq)]
        for df, (*_, seq) in zip(runs, selections)
    ]
    # [print(df.shape) for df in runs]
    return pd.concat(runs, axis=0).reset_index(drop=True)
