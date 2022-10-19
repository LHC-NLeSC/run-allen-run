#!/bin/bash

pushd /project/bfys/suvayua/codebaby/run-allen-run || exit

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments list 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100
python ./scanprops.py ../Allen/build/Sequence.json \
       --experiment-name v100 \
       --max-batch-size 256 2000 \
       --use-fp16
