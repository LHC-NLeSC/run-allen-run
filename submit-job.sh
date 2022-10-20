#!/bin/bash

#PBS -l nodes=1:v100
#PBS -l walltime=3:00:00
#PBS -l mem=4gb

pushd /project/bfys/suvayua/codebaby/run-allen-run || exit

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments list 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100

python ./scanprops.py ../Allen/build/Sequence.json \
       --experiment-name v100 \
       --batch-size-range 256 2000 \
       --fp16
