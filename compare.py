#!/usr/bin/env python
import ast
from io import StringIO
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


keep_t = {
    "status": "category",
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
    "tags.mlflow.runName": "string",
}

keep = ["start_time", *keep_t]

fill_vals = {
    "params.copies": 1,
    "params.onnx_input": "ghost_nn",
    "params.no_infer": False,
    "params.use_fp16": False,
    "params.use_int8": False,
}

dbg_cols = [
    "params.copies",
    "params.max_batch_size",
    "params.onnx_input",
    "params.no_infer",
    "tags.branch",
    "params.max_batch_size",
]


def df_round_trip(df: pd.DataFrame) -> pd.DataFrame:
    """Round trip the dataframe through CSV to convert the columns to correct dtypes"""
    return pd.read_csv(StringIO(df.to_csv(index=False)))


def process_df(df) -> pd.DataFrame:
    # dates = ("start_time", "end_time")
    # for col in dates:
    #     if col in df.columns:
    #         df[col] = pd.to_datetime(df[col])

    df["params.block_dim"] = df["params.block_dim"].map(
        lambda x: ast.literal_eval(x)[0]
    )
    df = df.fillna(fill_vals)
    if "params.onnx_input" in df.columns:
        df["params.onnx_input"] = df["params.onnx_input"].map(lambda p: Path(p).stem)
    else:
        # handcoded algorithm is based on the default model (ghost_nn)
        df.loc[:, "params.onnx_input"] = fill_vals["params.onnx_input"]

    df = df.astype({k: v for k, v in keep_t.items() if k in df.columns})
    # add categories now to avoid NaN later
    df["tags.branch"] = df["tags.branch"].cat.add_categories(["baseline", "handcoded"])
    return df.loc[:, [col for col in keep if col in df.columns]]


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
    baseline_copy = pd.concat(dfs, axis=0)
    baseline_copy.loc[:, "params.use_fp16"] = True
    dfs.append(baseline_copy)

    # main benchmarks
    dfs.append(ghostbuster[~no_infer_idx])

    # handcoded
    handcoded.loc[:, "tags.branch"] = "handcoded"
    max_block_dim = handcoded["params.block_dim"].max()  # pick any value
    dfs.append(handcoded[handcoded["params.block_dim"] == max_block_dim])
    return (
        pd.concat(dfs, axis=0).reset_index(drop=True).fillna(fill_vals).astype(keep_t)
    )


def get_facets(df: pd.DataFrame, use_fp16: bool = False, use_int8: bool = False):
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
