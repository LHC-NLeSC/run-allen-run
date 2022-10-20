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
