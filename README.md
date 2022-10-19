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

Setup on subsequent runs
```
$ source lhcb-setup.sh
$ source venv/bin/activate
```

Deactivate the environment with
```
$ deactivate
```

## Results

Run the mlflow UI server
```
$ mlflow ui
```
