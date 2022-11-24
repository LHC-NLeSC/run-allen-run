#!/bin/bash

# exclusive node
#PBS -n
#PBS -l nodes=1:v100
#PBS -l walltime=24:00:00
#PBS -l mem=4gb

echo "Job: ${PBS_JOBID}" |& tee current.log

pushd /project/bfys/$USER/codebaby/run-allen-run || exit

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments list 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100

if [[ -d Allen-master ]]; then
    echo "Job (master): hlt1_pp_default_seq.json" |& tee -a current.log
    python ./scanprops.py Allen-master/build/hlt1_pp_default_seq.json \
	   --experiment-name v100
else
    echo "Did you run ./build.sh & ./prepare.sh?" |& tee -a current.log
    echo "No build of master branch found" |& tee -a current.log > /dev/stderr
fi

if [[ -d Allen-ghostbuster ]]; then
    for i in {1..5}; do
	for f in nn nn_big; do
	    echo "Job (ghostbuster): ghostbuster_test_n${i}_seq.json" |& tee -a current.log
	    python ./scanprops.py Allen-ghostbuster/build/ghostbuster_test_n${i}_seq.json \
		   --experiment-name v100 \
		   --batch-size-range 256 16000 \
		   --no-infer \
		   --onnx-input /project/bfys/$USER/codebaby/Allen/input/ghost_${f}.onnx \
		   --copies $i
	done
    done
    echo "Job (ghostbuster): hlt1_pp_default_seq.json" |& tee -a current.log
    python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
	   --experiment-name v100
else
    echo "Did you run ./build.sh & ./prepare.sh?" |& tee -a current.log
    echo "No build of ghostbuster branch found" |& tee -a current.log > /dev/stderr
fi
