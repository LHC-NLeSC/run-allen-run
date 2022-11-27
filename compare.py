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

    # df["copies"] = df.copies.astype("Int64")
    df["onnx_input"] = df.onnx_input.dropna().map(lambda p: Path(p).stem)
    # .astype("string")

    df = df.drop(columns=["input_name", "use_fp16"])
    cols = ~np.array(
        [
            df.iloc[:, i].name == "duration" and isinstance(df.iloc[0, i], str)
            for i in range(len(df.columns))
        ]
    )
    df = df.iloc[:, cols]
    return df


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
    baseline = df[df.sequence.str.contains("hlt1")][["branch", "event_rate"]]
    ghostbuster = df[df.sequence.str.startswith("ghostbuster")]
    max_copies = int(ghostbuster.copies.max())

    # plt.ion()

    facets = sns.relplot(
        data=ghostbuster,
        x="max_batch_size",
        y="event_rate",
        hue="copies",
        palette=sns.color_palette("colorblind")[:max_copies],
        style="no_infer",
        dashes=False,
        markers=True,
        col="onnx_input",
        kind="line",
    )
    facets.refline(y=baseline.iloc[0, 1], label=baseline.iloc[0, 0], linestyle="--")
    facets.refline(y=baseline.iloc[1, 1], label=baseline.iloc[1, 0], linestyle=":")

    # FIXME: add the baseline legend next to the others
    plt.legend()
    facets.savefig("evt-rate-vs-batch-size-comparison.png")
