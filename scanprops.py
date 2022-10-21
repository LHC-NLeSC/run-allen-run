#!/usr/bin/env python
"""Scan algorithm parameters for benchmarking

"""

from itertools import product
import json
from math import log2
import os
from pathlib import Path
import re
from typing import Union

import mlflow
import sh

TEST_CMD = ["../foo.sh", "42 {config} --foo bar"]

ALLEN_CMD = [
    "../toolchain/wrapper",
    "../Allen -g ../../input/detector_configuration/ "
    "-t 12 --events-per-slice 1000 -n 1000 -r 100 "
    "--run-from-json 1 --sequence {config} "
    "--mdf /data/bfys/raaij/upgrade/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf",
]


NUMBERS = re.compile("[0-9]+.[0-9]+")

ENV = os.environ.copy()


def shtrip(output: Union[sh.RunningCommand, None]) -> str:
    """Strip shell stdout of newlines & whitespace"""
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


def param_matrix(batch_size_range: tuple[int, int], fp16: bool):
    """Create all permutations from a range of batch sizes, and fp16

    Parameters
    ----------
    batch_size_range: tuple[int, int]

      The range limits are rounded up to the nearest power of 2, and then
      intermediate powers of two are filled in.

    fp16: bool

      If True, permutation also includes fp16 support set to True or False, it
      is set to False otherwise

    Returns
    -------
    product[tuple[int, bool]]

    """
    lo, hi = [round_up_2(i) for i in batch_size_range]
    batch_sizes = (1 << i for i in range(int(log2(lo)), int(log2(hi) + 1)))
    return product(batch_sizes, (True, False) if fp16 else (fp16,))


def runner(config_json: Path, max_batch_size: int, use_fp16: bool) -> dict[str, float]:
    """Run Allen with the provided parameters, and log the metrics w/ mlflow

    Log files are saved in the run directory:
    - configuration: config-<parameters>.json
    - stdout: stdout-<parameters>.log
    - parameters & metrics: meta-<parameters>.json

    Parameters
    ----------
    config_json: Path

      Path to the JSON configuration to use as template

    max_batch_size: int

      Maximum batch size for our algorithm ("GhostProbabilityNN")

    use_fp16: bool

      Whether to enable FP16 optimisation

    Returns
    -------
    dict[str, float]

      Metrics: {"event_rate": 123.456, "duration": 123.456}

    """
    params = {"max_batch_size": max_batch_size, "use_fp16": use_fp16}
    mlflow.log_params(params)
    fname_part = f"batch-size-{max_batch_size}-fp16-{use_fp16}"

    rundir = config_json.parent / Path(f"run-{fname_part}")
    rundir.mkdir(exist_ok=True)
    config_edited = rundir / f"config-{fname_part}.json"

    config = json.loads(config_json.read_text())
    config["GhostProbabilityNN"]["max_batch_size"] = max_batch_size
    config["GhostProbabilityNN"]["use_fp16"] = use_fp16
    config_edited.write_text(json.dumps(config, indent=4))

    with sh.cd(rundir):
        env = ENV.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd, opts = ALLEN_CMD
        stdout = shtrip(
            sh.Command(cmd)(*opts.format(config=config_edited.name).split(), _env=env)
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
    mlflow.log_artifact(f"{rundir/log_file}")
    mlflow.log_artifact(f"{rundir/meta_file}")
    return metrics


def mlflow_run(expt_name: str, config_json: str, max_batch_size: int, use_fp16: bool):
    """Multiprocessing friendly wrapper to start an mlflow run"""
    expts = mlflow.search_experiments(filter_string=f"name = {expt_name!r}")
    if not expts:
        print("Couldn't find experiment, please setup using CLI")
        return {}
    expt_id = expts[0].experiment_id
    # causes race condition when using multiprocessing
    # expt_id = mlflow.create_experiment(expt_name)

    # FIXME: add git metadata in tags
    with mlflow.start_run(experiment_id=expt_id, tags={"branch": "ghostbuster"}):
        return runner(Path(config_json), max_batch_size, use_fp16)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_json", help="Config JSON, parent directory should have the binary"
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--batch-size-range", nargs=2, type=int)
    parser.add_argument("--fp16", action="store_true", help="Benchmark FP16 support")
    opts = parser.parse_args()
    params = tuple(param_matrix(opts.batch_size_range, opts.fp16))

    for batch, fp16 in param_matrix(opts.batch_size_range, opts.fp16):
        par, metric = mlflow_run(opts.experiment_name, opts.config_json, batch, fp16)
        print(par, metric)
