#!/bin/bash

# exclusive node
#PBS -n
#PBS -l nodes=1:v100
#PBS -l walltime=6:00:00
#PBS -l mem=4gb

pushd /project/bfys/$USER/codebaby/run-allen-run || exit

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments list 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100

if [[ -d Allen-master ]]; then
    python ./scanprops.py Allen-master/build/hlt1_pp_default_seq.json \
	   --experiment-name v100
else
    echo "Did you run ./build.sh?"
    echo "No build of master branch found" > /dev/stderr
fi

if [[ -d Allen-ghostbuster ]]; then
    python ./scanprops.py Allen-ghostbuster/build/ghostbuster_test_seq.json \
	   --experiment-name v100 \
	   --batch-size-range 256 16000 \
	   --no-infer
    python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
	   --experiment-name v100
else
    echo "Did you run ./build.sh?"
    echo "No build of ghostbuster branch found" > /dev/stderr
fi
