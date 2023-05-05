#!/usr/bin/env python
import ast
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

rename = {"Duration": "job_duration"}
# convert to seconds: pd.to_timedelta(df["job_duration"]).dt.total_seconds()

keep_t = {
    "params.copies": int,
    "params.block_dim": int,
    "params.max_batch_size": int,
    "params.no_infer": bool,
    "params.onnx_input": "category",
    "params.sequence": "string",
    "params.use_fp16": bool,
    "params.use_int8": bool,
    "metrics.duration": float,
    "metrics.event_rate": float,
    "tags.branch": "category",
    "tags.mlflow.runname": "string",
}

keep = ["start_time", *keep_t]

dbg_cols = [
    "params.copies",
    "params.max_batch_size",
    "params.onnx_input",
    "params.no_infer",
    "tags.branch",
    "params.max_batch_size",
]


def get_df(csv, before: str = "", after: str = "") -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(csv, index_col=0)).rename(columns=rename)
    df.columns = df.columns.str.replace(" ", "_").str.casefold()

    queries = ["status == 'FINISHED'"]  # drop failed/incomplete jobs
    if before:
        queries += [f"start_time < {before!r}"]
    if after:
        queries += [f"start_time > {after!r}"]
    query = "&".join([f"({q})" for q in queries])
    df = df.query(query).copy()

    if "params.block_dim" in df.columns:
        df["params.block_dim"] = df["params.block_dim"].map(
            lambda x: ast.literal_eval(x)[0]
        )

    dates = ["start_time"]
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    bools = ["params.no_infer", "params.use_fp16", "params.use_int8"]
    for col in bools:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(False)

    if all(map(lambda i: i in df.columns, ("params.copies", "params.onnx_input"))):
        # copies is 1 by default
        df["params.copies"] = df["params.copies"].fillna(1).infer_objects()
        # onnx_input is missing for hand coded, but it is based on ghost_nn
        df["params.onnx_input"] = (
            df["params.onnx_input"].fillna("ghost_nn").map(lambda p: Path(p).stem)
        )

    return df.loc[:, keep].convert_dtypes()


def df_extend_if(ghostbuster: pd.DataFrame, handcoded: pd.DataFrame) -> pd.DataFrame:
    """Extend the dataframe from ghostbuster jobs for easier plotting"""

    def extend_batch(row, sizes, onnx=None):
        res = pd.concat([row] * len(sizes), axis=1).T
        res["params.max_batch_size"] = sizes
        return res.infer_objects()

    # baseline
    no_infer_idx = ghostbuster["params.no_infer"]
    single_copy_idx = ghostbuster["params.copies"] == 1
    baseline_idx = no_infer_idx & single_copy_idx
    ghostbuster.loc[baseline_idx, "tags.branch"] = "baseline"
    batch_sizes = ghostbuster["params.max_batch_size"].unique()
    dfs = [
        extend_batch(row, batch_sizes)
        for _, row in ghostbuster[baseline_idx].iterrows()
    ]
    baseline_copy = ghostbuster[baseline_idx].copy()
    baseline_copy.loc[:, "params.use_fp16"] = True
    dfs.extend(extend_batch(row, batch_sizes) for _, row in baseline_copy.iterrows())

    # main benchmarks
    dfs.append(ghostbuster[~no_infer_idx])

    # handcoded
    handcoded.loc[:, "tags.branch"] = "handcoded"
    max_block_dim = handcoded["params.block_dim"].max()  # pick any value
    dfs.append(handcoded[handcoded["params.block_dim"] == max_block_dim])
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def get_facets(df: pd.DataFrame, use_fp16: bool = False, use_int8: bool = False):
    df = df.astype(keep_t)
    df.info()
    if use_int8 and use_fp16:
        raise ValueError("Cannot use both fp16 and int8")
    elif use_fp16 and not use_int8:
        df = df[~df["params.use_int8"]]
        facet_col = {"col": "params.use_fp16"}
    elif use_int8 and not use_fp16:
        df = df[~df["params.use_fp16"]]
        facet_col = {"col": "params.use_int8"}
    else:
        df = df[~df["params.use_int8"] & ~df["params.use_fp16"]]
        facet_col = {}

    if all(map(lambda i: i in df.columns, ("params.copies", "params.onnx_input"))):
        max_copies = int(df["params.copies"].max())
        add_opts = {
            "hue": "params.copies",
            "palette": sns.color_palette("colorblind")[:max_copies],
            "row": "params.onnx_input",
            **facet_col,
        }
    else:
        add_opts = {}

    sns.set_style("darkgrid")
    facets = sns.relplot(
        data=df,
        x="params.max_batch_size",
        y="metrics.event_rate",
        style="tags.branch",
        dashes=False,
        markers=True,
        kind="line",
        **add_opts,
    )
    # facets.refline(y=91000, label="ref", linestyle="--")
    # facets.refline(y=baseline.iloc[1, 1], label=baseline.iloc[1, 0], linestyle=":")
    return facets


if __name__ == "__main__":
    import argparse

    doc = "Runs maybe filtered on date using --before/after"
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("csv", help="CSV file with run statistics")
    parser.add_argument(
        "--after",
        default="",
        help="Datetime string (%%Y-%%m-%%d %%H:%%M:%%s) to filter runs",
    )
    parser.add_argument(
        "--before",
        default="",
        help="Datetime string (%%Y-%%m-%%d %%H:%%M:%%s) to filter runs",
    )
    opts = parser.parse_args()

    df = get_df(opts.csv, after=opts.after, before=opts.before)
    ghostbuster = df[df["params.sequence"].str.startswith("ghostbuster_")]
    ghostbusterhc = df[df["params.sequence"].str.startswith("ghostbusterhc_")]
    df_plot = df_extend_if(ghostbuster, ghostbusterhc)

    facets = get_facets(df_plot, use_fp16=True)
    facets.savefig("evt-rate-vs-batch-size-comparison.png")
