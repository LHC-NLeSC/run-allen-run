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
    for i in {1..4};
    do
	for f in nn nn_big;
	do
	    python ./scanprops.py Allen-ghostbuster/build/ \
		   --experiment-name v100 \
		   --batch-size-range 256 16000 \
		   --no-infer \
		   --onnx-input /project/bfys/suvayua/codebaby/Allen/input/ghost_${f}.onnx \
		   --copies $i
	done
    done
    python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
	   --experiment-name v100
else
    echo "Did you run ./build.sh?"
    echo "No build of ghostbuster branch found" > /dev/stderr
fi
