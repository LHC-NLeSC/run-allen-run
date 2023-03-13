# Run Allen for benchmarking

Run Allen with different algorithm properties

## Setup

Intial setup
```
$ source lhcb-setup.sh
$ python -m virtualenv --prompt run-allen venv
$ source venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

Setup for subsequent runs
```
$ source lhcb-setup.sh
$ source venv/bin/activate
```

### Allen application setup

The Allen application needs to be run from JSON configuration.  So
first you need to write the configuration to a JSON file by running
with `--write-configuration 1`.  Subsequently, we can run the
application from the JSON configuration with the option
`--run-from-json 1 --sequence /path/to/config.json`.

The application accesses some files in the current directory, so this
script separates the running directory for each run.

The repo provides the scripts [`build.sh`](./build.sh) and
[`prepare.sh`](./prepare.sh) that makes this process simpler.

- `build.sh`: checks out and builds the specified branch from a local
  Allen repository in a separate directory under the current
  directory.
  ```
  $ ls -d Allen*
  Allen-ghostbuster  Allen-master
  ```

- `prepare.sh`: writes out the JSON configuration such that the job
  scripts [`scanprops.py`](./scanprops.py) and
  [`submit-job.sh`](./submit-job.sh) can be run.

### Cleanup

Deactivate the environment with
```
$ deactivate
```

## Results

Run the mlflow UI server
```
$ mlflow ui
```

## Notes on profiling

```
$ ./toolchain/wrapper ncu --set full -o <profile-output> <allen-cmd>
$ ./toolchain/wrapper nsys profile -t cuda,nvtx,cudnn,cublas -o <profile-output> <allen-cmd>
```

## Adding a new option
To add a new option, the following functions need to be updated:
- add a corresponding property in the `scanprops.jobopts_t` dataclass
  - update `jobopts_t.fname_part`
- `scanprops.param_matrix`: ensures the option is included in all
  permutations
- `scanprops.get_config`: update the docstring
- CLI argument parser
- `jobopt_t` instance in the entry point
- the for-loop over `param_matrix` in the entry point
- add option in submit-job.sh
- update `jobopts_t` dataclass instance in `genconf.py`
