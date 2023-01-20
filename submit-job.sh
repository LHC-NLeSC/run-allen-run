#!/bin/bash

# exclusive node
#PBS -n
#PBS -l nodes=1:v100
#PBS -l walltime=96:00:00
#PBS -l mem=4gb

pushd /project/bfys/$USER/codebaby/run-allen-run || exit

echo "Job: ${PBS_JOBID}" |& tee current.log

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments search 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100

if [[ -d Allen-ghostbuster ]]; then
    echo "Jobs w/ ghostbuster:" |& tee -a current.log
    cp -v ./allen_benchmarks.py Allen-ghostbuster/configuration/python/AllenCore/ |& tee -a current.log
    for i in {1..5}; do
	for f in nn_tiny nn nn_big nn_bigger; do
	    echo "Job (ghostbuster): ghostbuster_test_n${i}_seq.json" |& tee -a current.log
	    python ./scanprops.py Allen-ghostbuster/build/ghostbuster_test_n${i}_seq.json \
		   --experiment-name v100 \
		   --batch-size-range 256 16000 \
		   --fp16 \
		   --no-infer \
		   --onnx-input /project/bfys/$USER/codebaby/Allen/input/ghost_${f}.onnx \
		   --copies $i |& tee -a current.log
	done
    done
    echo "Job (ghostbuster): hlt1_pp_default_seq.json" |& tee -a current.log
    python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
	   --experiment-name v100 |& tee -a current.log
else
    echo "Did you run ./build.sh & ./prepare.sh?" |& tee -a current.log
    echo "No build of ghostbuster branch found" |& tee -a current.log > /dev/stderr
fi
