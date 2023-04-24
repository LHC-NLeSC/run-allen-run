#!/usr/bin/env python
"""Scan algorithm parameters for benchmarking

"""

from dataclasses import asdict, dataclass
from itertools import chain, product
import json
from math import log2
import os
from pathlib import Path
import re
from typing import Iterable, Union

import mlflow
import onnx
import sh

TEST_CMD = ["../foo.sh", "42 {config} --foo bar"]

ALLEN_CMD = [
    "../toolchain/wrapper",
    "../Allen -g ../../input/detector_configuration/ "
    "-t 12 --events-per-slice 1000 -n 1000 -r 100 "
    "{flag} --sequence {sequence} "
    "--mdf /data/bfys/raaij/upgrade/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf",
]

JSON_FLAG = "--run-from-json 1"

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


def expand_range_2(start: int, stop: int) -> list[int]:
    """Expand a range of numbers to a list of powers of 2"""
    lo, hi = [round_up_2(i) for i in (start, stop)]
    return [1 << i for i in range(int(log2(lo)), int(log2(hi) + 1))]


def param_matrix(
    batch_size_range: tuple[int, int], no_infer: bool, fp16: bool, int8: bool
) -> Iterable[tuple[int, bool, bool, bool]]:
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

    int8: bool

      If True, permutation also includes int8 support set to True or False, it
      is set to False otherwise

    Returns
    -------
    Iterable[tuple[int, bool, bool, bool]]

    """
    batch_sizes = expand_range_2(*batch_size_range)
    fp16_opts = (True, False) if fp16 else (False,)
    int8_opts = (True, False) if int8 else (False,)
    perms = (
        opts
        for opts in product(batch_sizes, (False,), fp16_opts, int8_opts)
        if not (opts[2] is True and opts[3] is True)
    )
    if no_infer:
        # w/ no inference, doesn't matter if FP16 is enabled
        return chain(perms, product([batch_sizes[-1]], (True,), (False,), (False,)))
    else:
        return perms


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


def git_diff_stat() -> list[str]:
    return [
        line.rsplit()[-1] for line in sh.git("diff-index", "HEAD").strip().split("\n")
    ]


def git_root() -> Path:
    return Path(shtrip(sh.git("rev-parse", "--show-toplevel")))


def mlflow_src_branch() -> str:
    run = mlflow.active_run()
    assert run is not None, "runner: no active run, something went wrong"
    return run.data.tags["branch"]


@dataclass
class jobopts_t:
    max_batch_size: int
    onnx_input: str
    input_name: str
    no_infer: bool = False
    use_fp16: bool = False
    use_int8: bool = False
    block_dim: tuple[int, int, int] = (256, 1, 1)
    copies: int = 1

    @property
    def not_props(self):
        return ["copies"]

    @property
    def fname_part(self):
        """File path friendly string, encoded w/ parameter values"""
        params = asdict(self)
        params["onnx_input"] = Path(self.onnx_input).stem
        fname_part = "batch-size-{max_batch_size}-"
        fname_part += "no-infer-{no_infer}-"
        fname_part += "fp16-{use_fp16}-"
        fname_part += "int8-{use_int8}-"
        fname_part += "block-dim-{block_dim[0]}-"
        fname_part += "onnx-{onnx_input}"
        return fname_part.format(**params)


def onnx_input_name(onnx_input: str):
    """Return the input array name for the ONNX file"""
    return onnx.load(onnx_input).graph.input[0].name


def get_config(config: dict, opts: jobopts_t) -> dict:
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
      - use_int8: whether to enable INT8 optimisation

    Returns
    -------
    dict

      JSON encoded configuration

    """
    params = asdict(opts)
    mlflow.log_params(params)
    for i in range(opts.copies):
        config[f"GhostProbabilityNN{i}"].update(
            (k, v) for k, v in params.items() if k not in opts.not_props
        )
    return config


def write_config_json(
    builddir: Path, fname: str, config: dict, opts: jobopts_t
) -> tuple[Path, Path]:
    if opts.max_batch_size < 0:  # ghostbuster algorithm not included in sequence
        fname_part = mlflow_src_branch()
    else:
        fname_part = opts.fname_part
        config = get_config(config, opts)

    rundir = builddir / Path(f"run-{fname_part}")
    rundir.mkdir(exist_ok=True)
    config_edited = rundir / f"{fname}-{fname_part}.json"
    config_edited.write_text(json.dumps(config, indent=4))
    return rundir, config_edited


def runner(rundir: Path, config_edited: Path, jobopts: jobopts_t) -> dict[str, float]:
    """Run Allen with the configuration, and log the metrics w/ mlflow

    Log files are saved in the run directory (logged as artifacts w/ mlflow):
    - configuration: config-<parameters>.json
    - stdout: stdout-<parameters>.log
    - parameters & metrics: meta-<parameters>.json

    Parameters
    ----------
    rundir: Path

    config_edited: Path

      Path to the JSON/Python configuration

    jobopts: jobopts_t

      Job options (parameter values), included in metadata log

    Returns
    -------
    dict[str, float]

      Metrics: {"event_rate": 123.456, "duration": 123.456}

    """
    if jobopts.max_batch_size < 0:  # ghostbuster algorithm not included in sequence
        fname_part = mlflow_src_branch()
    else:
        fname_part = jobopts.fname_part

    flag = JSON_FLAG
    sequence = config_edited.relative_to(rundir)
    mlflow.log_param("sequence", sequence.stem)

    params = asdict(jobopts)
    params.update(sequence=str(sequence))

    with sh.cd(rundir):
        env = ENV.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd, opts = ALLEN_CMD
        stdout = shtrip(
            sh.Command(cmd)(
                *opts.format(flag=flag, sequence=sequence).split(), _env=env
            )
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


def mlflow_run(expt_name: str, path: str, opts: jobopts_t):
    """Wrapper to start an mlflow run"""
    expts = mlflow.search_experiments(filter_string=f"name = {expt_name!r}")
    if not expts:
        print("Couldn't find experiment, please setup using CLI")
        return {}
    expt_id = expts[0].experiment_id
    # causes race condition when using multiprocessing
    # expt_id = mlflow.create_experiment(expt_name)

    _path = Path(path)
    builddir = _path.parent
    config = json.loads(_path.read_text())

    if opts.max_batch_size > 0:  # ghostbuster algorithm in sequence
        copies = len([k for k in config.keys() if k.startswith("GhostProbabilityNN")])
        assert (
            copies == opts.copies
        ), f"number of copies don't match in base config: '{copies=}'!='{opts.copies=}'"

    with sh.cd(builddir):
        tags = asdict(git_commit())
        if tags["dirty"]:
            tags["dirty_files"] = git_diff_stat()
        print(tags)

    with mlflow.start_run(experiment_id=expt_id, tags=tags):
        rundir, config_edited = write_config_json(builddir, _path.stem, config, opts)
        return runner(rundir, config_edited, opts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_json",
        help="Config JSON, parent directory should have the binary",
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--batch-size-range", nargs=2, type=int)
    parser.add_argument("--no-infer", action="store_true", help="Toggle inference")
    parser.add_argument("--fp16", action="store_true", help="Benchmark FP16 support")
    parser.add_argument("--int8", action="store_true", help="Benchmark INT8 support")
    parser.add_argument(
        "--onnx-input", help="Path to ONNX file that should be used for inference"
    )
    parser.add_argument(
        "--copies", type=int, default=1, help="Number of algo instances"
    )
    parser.add_argument(
        "--block-dim-range", nargs=2, type=int, help="Block dimension range"
    )

    opts = parser.parse_args()
    jobopts = jobopts_t(
        max_batch_size=-1,  # dummy
        no_infer=opts.no_infer,
        use_fp16=opts.fp16,
        use_int8=opts.int8,
        onnx_input=opts.onnx_input,
        input_name=onnx_input_name(opts.onnx_input),
        copies=opts.copies,
    )

    if opts.batch_size_range is None:
        # dummy parameter values, they are ignored when ghostbuster isn't included
        jobopts.max_batch_size = -1
        metric = mlflow_run(opts.experiment_name, opts.config_json, jobopts)
        print(metric)
    else:
        for batch, no_infer, fp16, int8 in param_matrix(
            opts.batch_size_range, opts.no_infer, opts.fp16, opts.int8
        ):
            jobopts.max_batch_size = batch
            if isinstance(jobopts, jobopts_t):  # onnx
                jobopts.no_infer = no_infer
                jobopts.use_fp16 = fp16
                jobopts.use_int8 = int8
            if opts.block_dim_range is None:
                metric = mlflow_run(opts.experiment_name, opts.config_json, jobopts)
                print(metric)
            else:
                for block_dim in expand_range_2(*opts.block_dim_range):
                    jobopts.block_dim = (block_dim, 1, 1)
                    metric = mlflow_run(opts.experiment_name, opts.config_json, jobopts)
                    print(metric)
