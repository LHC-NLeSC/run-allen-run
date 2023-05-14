#!/usr/bin/env python
"""Select runs from mlflow datastore using a TOML config file

      [[runs.<name>]]
      start = "2023-04-24 12:00"
      stop = "2023-04-30 23:00"
      seq = "<sequence>"

Match against sequence name as `<sequence>*`

"""

from argparse import ArgumentParser
import ast
from io import StringIO
from pathlib import Path

import seaborn as sns
from numpy.testing import assert_
import pandas as pd

from scanprops import RawArgDefaultFormatter
from selections import read_config_toml, select_runs

col_t = {
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
    "optimisation": "category",
}

keep = ["start_time", *[c for c in col_t if "use_" not in c]]

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


def df_col_t(df) -> dict[str, type | str]:
    return {k: v for k, v in col_t.items() if k in df.columns}


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

    df = df.astype(df_col_t(df))

    # encode optimisation flags as category
    if any("use_" in col for col in df.columns):
        opt_one_hot_encoded = pd.concat(
            [
                df["params.use_fp16"].rename("fp16"),
                df["params.use_int8"].rename("int8"),
                (~df["params.use_fp16"] & ~df["params.use_int8"]).rename("none"),
            ],
            axis=1,
        )
        opt = pd.from_dummies(opt_one_hot_encoded).astype("category").iloc[:, 0]
        df = df.assign(optimisation=opt)
    else:
        df = df.assign(optimisation="none")

    # add categories to branch now to avoid NaN later
    df["tags.branch"] = df["tags.branch"].cat.add_categories(["baseline", "handcoded"])

    return df.loc[:, [col for col in keep if col in df.columns]]


def df_extend_if(
    ghostbuster: pd.DataFrame, handcoded: pd.DataFrame, flag: str
) -> pd.DataFrame:
    """Extend the dataframe from ghostbuster jobs for easier plotting"""
    assert_(flag in ("fp16", "int8", "none", "both"))

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

    def extend_optimisation(df, flag):
        df.loc[:, "optimisation"] = flag
        return df

    if flag != "none":
        _baseline = pd.concat(dfs, axis=0)
        if flag in ("both", "fp16"):
            dfs.append(extend_optimisation(_baseline.copy(), "fp16"))
        if flag in ("both", "int8"):
            dfs.append(extend_optimisation(_baseline.copy(), "int8"))

    # main benchmarks
    dfs.append(ghostbuster[~no_infer_idx])

    # handcoded
    handcoded.loc[:, "tags.branch"] = "handcoded"
    max_block_dim = handcoded["params.block_dim"].max()  # pick any value
    dfs.append(handcoded[handcoded["params.block_dim"] == max_block_dim])
    df = pd.concat(dfs, axis=0).reset_index(drop=True).fillna(fill_vals)
    return df.astype(df_col_t(df))


def get_facets(ghostbuster: pd.DataFrame, handcoded: pd.DataFrame, flag: str):
    flags = {"fp16", "int8", "none", "both"}
    assert_(flag in flags)
    df = df_extend_if(ghostbuster, handcoded, flag)

    if flag != "both":
        df = df[df["optimisation"] == flag]
        df.loc[:, "optimisation"] = df["optimisation"].cat.remove_unused_categories()
    facet_col = {} if flag == "none" else {"col": "optimisation"}

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
    parser = ArgumentParser(description=__doc__, formatter_class=RawArgDefaultFormatter)
    parser.add_argument("toml_config", help="TOML file with run selection")
    parser.add_argument(
        "--flag",
        choices=["both", "fp16", "int8", "none"],
        default="both",
        help="Plot runs with TensorRT optimisation flag",
    )
    opts = parser.parse_args()

    selections = read_config_toml(opts.toml_config)["runs"].items()
    ghostbuster, handcoded = [
        select_runs([sel[0].values()]).pipe(df_round_trip).pipe(process_df)
        for run, sel in selections
    ]

    facets = get_facets(ghostbuster, handcoded, opts.flag)
    suffix = f"-{opts.flag}" if opts.flag != "none" else ""
    facets.savefig(f"evt-rate-vs-batch-size-comparison{suffix}.png")
