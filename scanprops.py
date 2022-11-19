#!/usr/bin/env python
"""Scan algorithm parameters for benchmarking

"""

from dataclasses import asdict, dataclass
from itertools import product
import json
from math import log2
import os
from pathlib import Path
import re
from typing import Union

import mlflow
import onnx
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


def param_matrix(batch_size_range: tuple[int, int], no_infer: bool, fp16: bool):
    """Create all permutations from a range of batch sizes, no infer, and use fp16

    Parameters
    ----------
    batch_size_range: tuple[int, int]

      The range limits are rounded up to the nearest power of 2, and then
      intermediate powers of two are filled in.

    no_infer: bool

      If True, permutation also includes a run without any inference

    fp16: bool

      If True, permutation also includes fp16 support set to True or False, it
      is set to False otherwise

    Returns
    -------
    product[tuple[int, bool]]

    """
    lo, hi = [round_up_2(i) for i in batch_size_range]
    batch_sizes = (1 << i for i in range(int(log2(lo)), int(log2(hi) + 1)))
    return product(
        batch_sizes,
        (True, False) if no_infer else (False,),
        (True, False) if fp16 else (False,),
    )


@dataclass
class Source:
    branch: str
    commit: str  # hash
    date: str  # datetime
    dirty: bool


def git_commit() -> Source:
    """Git commit metadata"""
    # FIXME: include working directory dirty or not
    branch = shtrip(sh.git.branch("--show-current"))
    return Source(
        branch=branch,
        commit=shtrip(sh.git("show-ref", "--hash=9", f"refs/heads/{branch}")),
        date=shtrip(
            sh.git.log(
                "--date=iso8601-strict", "--format=%ad", "-1", branch, _tty_out=False
            )
        ),
        dirty=bool(sh.git("diff-index", "--shortstat", "HEAD")),
    )


@dataclass
class jobopts_t:
    max_batch_size: int
    onnx_input: str
    input_name: str
    no_infer: bool = False
    use_fp16: bool = False
    copies: int = 1


def get_config(config: dict, opts: jobopts_t) -> tuple[dict, str, dict]:
    """Get config with new parameter values, and log them w/ mlflow

    Also encode the parameters in a string to be used in file names.

    Parameters
    ----------
    config: dict

      JSON configuration to use as template

    opts: jobopts_t

      Job options: only the following attributes are used
      - batch_size: maximum batch size for our algorithm ("GhostProbabilityNN")
      - no_infer: whether to disable inference for benchmarking
      - use_fp16: whether to enable FP16 optimisation

    Returns
    -------
    tuple[dict, str, dict]

      dict: full configuration, str: string encoded w/ parameter values, dict:
      just the params

    """
    onnx_input = Path(opts.onnx_input)
    params = asdict(opts)
    mlflow.log_params(params)

    fname_part = "batch-size-{max_batch_size}-"
    fname_part += "no-infer-{no_infer}-"
    fname_part += "fp16-{use_fp16}-"
    fname_part += "onnx-{onnx_input}"
    fname_part.format(**{**params, "onnx_input": onnx_input.stem})

    config["GhostProbabilityNN"].update((k, v) for k, v in params if k != "copies")
    return (config, fname_part, params)


def runner(
    config_json: Path, config: dict, fname_part: str, params: dict
) -> dict[str, float]:
    """Run Allen with the configuration, and log the metrics w/ mlflow

    Log files are saved in the run directory (logged as artifacts w/ mlflow):
    - configuration: config-<parameters>.json
    - stdout: stdout-<parameters>.log
    - parameters & metrics: meta-<parameters>.json

    Parameters
    ----------
    config_json: Path

      Path to the template JSON configuration, used to determine run directory

    config: dict

      Allen job configuration

    fname_part: str

      Unique string to use in filenames

    params: dict

      Dict with parameter values, included in metadata log

    Returns
    -------
    dict[str, float]

      Metrics: {"event_rate": 123.456, "duration": 123.456}

    """
    rundir = config_json.parent / Path(f"run-{fname_part}")
    rundir.mkdir(exist_ok=True)
    config_edited = rundir / f"config-{fname_part}.json"
    config_edited.write_text(json.dumps(config, indent=4))

    sequence = config_json.stem
    params.update(sequence=sequence)
    mlflow.log_param("sequence", sequence)

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


def mlflow_run(expt_name: str, config_json: str, opts: jobopts_t):
    """Multiprocessing friendly wrapper to start an mlflow run"""
    expts = mlflow.search_experiments(filter_string=f"name = {expt_name!r}")
    if not expts:
        print("Couldn't find experiment, please setup using CLI")
        return {}
    expt_id = expts[0].experiment_id
    # causes race condition when using multiprocessing
    # expt_id = mlflow.create_experiment(expt_name)

    config_json_path = Path(config_json)
    with sh.cd(config_json_path.parent):
        tags = asdict(git_commit())
        print(tags)

    with mlflow.start_run(experiment_id=expt_id, tags=tags):
        config = json.loads(config_json_path.read_text())
        if opts.max_batch_size < 0:  # ghostbuster algorithm not included in sequence
            fname_part = tags["branch"]
            params = {}
        else:
            config, fname_part, params = get_config(config, opts)
        return runner(config_json_path, config, fname_part, params)


def onnx_input_name(onnx_input: str):
    return onnx.load(onnx_input).graph.input[0].name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_json", help="Config JSON, parent directory should have the binary"
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--batch-size-range", nargs=2, type=int)
    parser.add_argument("--no-infer", action="store_true", help="Toggle inference")
    parser.add_argument("--fp16", action="store_true", help="Benchmark FP16 support")
    parser.add_argument(
        "--onnx-input",
        default="/project/bfys/suvayua/codebaby/Allen/input/ghost_nn.onnx",
        help="Path to ONNX file that should be used for inference",
    )

    opts = parser.parse_args()
    jobopts = jobopts_t(
        max_batch_size=-1,  # dummy
        no_infer=opts.infer,
        use_fp16=opts.fp16,
        onnx_input=opts.onnx_input,
        input_name=onnx_input_name(opts.onnx_input),
    )

    if opts.batch_size_range is None:
        # dummy parameter values, they are ignored when ghostbuster isn't included
        jobopts.max_batch_size = -1
        metric = mlflow_run(opts.experiment_name, opts.config_json, jobopts)
        print(metric)
    else:
        for batch, no_infer, fp16 in param_matrix(
            opts.batch_size_range, opts.no_infer, opts.fp16
        ):
            jobopts.max_batch_size = batch
            jobopts.no_infer = no_infer
            jobopts.use_fp16 = fp16
            metric = mlflow_run(opts.experiment_name, opts.config_json, jobopts)
            print(metric)
