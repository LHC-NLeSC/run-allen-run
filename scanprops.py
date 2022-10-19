#!/usr/bin/env python
"""Scan algorithm parameters for benchmarking

"""

from concurrent.futures import ProcessPoolExecutor
from itertools import product, repeat
import json
from math import log2
from pathlib import Path
import re
from typing import Union

import mlflow
import sh

TEST_CMD = ["./foo.sh", "42 {config} --foo bar {batch} {fp16}"]

ALLEN_CMD = [
    "./toolchain/wrapper",
    "./Allen -t 12 --events-per-slice 1000 -n 1000 -r 100 "
    "--run-from-json 1 --sequence {config} "
    "--mdf /data/bfys/raaij/upgrade/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf",
]


NUMBERS = re.compile("[0-9]+.[0-9]+")


def shtrip(output: Union[sh.RunningCommand, None]) -> str:
    if output is None:
        return ""
    return repr(output).strip()


def round_up_2(val: int) -> int:
    """Round up to the nearest power of 2

    source: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    """
    val -= 1
    val |= val >> 1
    val |= val >> 2
    val |= val >> 4
    val |= val >> 8
    val |= val >> 16
    val += 1
    return val


def param_matrix(max_batch_size: tuple[int, int], use_fp16: bool):
    lo, hi = [round_up_2(i) for i in max_batch_size]
    batch_sizes = (1 << i for i in range(int(log2(lo)), int(log2(hi) + 1)))
    return product(batch_sizes, (True, False) if use_fp16 else (use_fp16,))


def runner(config_json: Path, max_batch_size: int, use_fp16: bool) -> dict[str, float]:
    # FIXME: get runid, log parameters
    params = {"max_batch_size": max_batch_size, "use_fp16": use_fp16}
    mlflow.log_params(params)
    fname_part = f"batch-size-{max_batch_size}-fp16-{use_fp16}"

    bindir = config_json.parent
    config_edited = bindir / f"config-{fname_part}.json"

    config = json.loads(config_json.read_text())
    config["GhostProbabilityNN"]["max_batch_size"] = max_batch_size
    config["GhostProbabilityNN"]["use_fp16"] = use_fp16
    config_edited.write_text(json.dumps(config, indent=4))

    with sh.cd(bindir):
        cmd, opts = ALLEN_CMD
        stdout = shtrip(
            sh.Command(cmd)(*opts.format(config=config_edited.name).split())
        )
        log_file = Path(f"stdout-{fname_part}.log")
        log_file.write_text(stdout)
        try:
            event_rate, duration = [
                float(match[0])
                for line in stdout.splitlines()[-2:]
                if (match := NUMBERS.search(line))
            ]
        except ValueError:
            event_rate, duration = -1.0, -1.0
        metrics = {"event_rate": event_rate, "duration": duration}
        meta_file = Path(f"meta-{fname_part}.json")
        meta_file.write_text(json.dumps({**params, **metrics}, indent=2))

    mlflow.log_metrics(metrics)
    mlflow.log_artifact(f"{config_edited}")
    mlflow.log_artifact(f"{bindir/log_file}")
    mlflow.log_artifact(f"{bindir/meta_file}")
    return metrics


def mlflow_run(expt_name, config_json, max_batch_size, use_fp16):
    expts = mlflow.search_experiments(filter_string=f"name = {expt_name!r}")
    if len(expts) > 0:
        expt_id = expts[0].experiment_id
    else:
        expt_id = mlflow.create_experiment(expt_name)

    with mlflow.start_run(experiment_id=expt_id, tags={"branch": "ghostbuster"}):
        return runner(config_json, max_batch_size, use_fp16)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_json", help="Config JSON, parent directory should have the binary"
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-batch-size", nargs=2, type=int, help="Batch size range")
    parser.add_argument(
        "--use-fp16", action="store_true", help="Benchmark FP16 support"
    )
    opts, rest = parser.parse_known_args()
    config_json = Path(opts.config_json)
    params = tuple(param_matrix(opts.max_batch_size, opts.use_fp16))
    njobs = len(params)

    with ProcessPoolExecutor(2) as executor:
        for par, metric in zip(
            params,
            executor.map(
                mlflow_run,
                repeat(opts.experiment_name),
                repeat(config_json, njobs),
                *zip(*params),
            ),
        ):
            print(par, metric)
