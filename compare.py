#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

drop_cols = [
    "source_name",
    "source_type",
    "user",
    "status",
    "dirty",
    "run_id",
    "commit",
    "date",
]


def get_df(csv, before: str = "", after: str = "") -> pd.DataFrame:
    df = pd.read_csv(csv)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df = df.drop(drop_cols, axis=1)

    queries = []
    if before:
        queries += [f"start_time < {before!r}"]
    if after:
        queries += [f"start_time > {after!r}"]
    query = "&".join([f"({q})" for q in queries])
    df = df.query(query).copy()

    bools = ["no_infer", "use_fp16"]
    for col in bools:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    dates = ["start_time"]
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if all(map(lambda i: i in df.columns, ("copies", "onnx_input"))):
        df["copies"] = df.copies.fillna(1).infer_objects()
        df["onnx_input"] = df.onnx_input.dropna().map(lambda p: Path(p).stem)
        # .astype("string")

    cols = ~np.array(
        [
            df.iloc[:, i].name == "duration" and isinstance(df.iloc[0, i], str)
            for i in range(len(df.columns))
        ]
    )
    df = df.iloc[:, cols]
    return df


def df_extend_if(ghostbuster: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Extend the dataframe from ghostbuster jobs for easier plotting"""

    def extend_batch(row, sizes):
        res = pd.concat([row] * len(sizes), axis=1).T
        res["max_batch_size"] = sizes
        return res.infer_objects()

    batch_sizes = ghostbuster.max_batch_size.unique()
    dfs = [ghostbuster.query("~no_infer")]

    df_ni = ghostbuster.query("no_infer")
    if len(df_ni.max_batch_size.unique()) == 1:
        dfs.extend(extend_batch(row, batch_sizes) for _, row in df_ni.iterrows())

    dfs.extend(extend_batch(row, batch_sizes) for _, row in baseline.iterrows())
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def get_facets(df: pd.DataFrame):
    if all(map(lambda i: i in df.columns, ("copies", "onnx_input"))):
        max_copies = int(df.copies.max())
        add_opts = {
            "hue": "copies",
            "palette": sns.color_palette("colorblind")[:max_copies],
            "row": "onnx_input",
            "col": "use_fp16",
        }
    else:
        add_opts = {}

    facets = sns.relplot(
        data=df,
        x="max_batch_size",
        y="event_rate",
        style="no_infer",
        dashes=False,
        markers=True,
        kind="line",
        **add_opts,
    )
    # facets.refline(y=baseline.iloc[0, 1], label=baseline.iloc[0, 0], linestyle="--")
    # facets.refline(y=baseline.iloc[1, 1], label=baseline.iloc[1, 0], linestyle=":")
    return facets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file with run statistics")
    parser.add_argument(
        "--after", default="", help="Datetime string (%Y-%m-%d %H:%M:%s) to filter runs"
    )
    parser.add_argument(
        "--before",
        default="",
        help="Datetime string (%Y-%m-%d %H:%M:%s) to filter runs",
    )
    opts = parser.parse_args()

    df = get_df(opts.csv, after=opts.after, before=opts.before)
    ghostbuster = df[df.sequence.str.startswith("ghostbuster")]
    baseline = df[df.sequence.str.contains("hlt1")]
    baseline["no_infer"] = baseline["branch"]
    df_plot = df_extend_if(ghostbuster, baseline)

    facets = get_facets(df_plot)
    facets.savefig("evt-rate-vs-batch-size-comparison.png")
