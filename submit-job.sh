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

[[ -d Allen-ghostbuster/build ]] || \
    {
	echo "No build of ghostbuster branch found" |& tee -a current.log > /dev/stderr
	echo "Did you run ./build.sh & ./prepare.sh?" |& tee -a current.log
	exit 1
    }

echo "Jobs w/ ghostbuster:" |& tee -a current.log
cp -v ./allen_benchmarks.py Allen-ghostbuster/configuration/python/AllenCore/ |& tee -a current.log

for i in {1..5}; do
    json_config=Allen-ghostbuster/build/ghostbuster_test_n${i}_seq.json
    [[ -f ${json_config} ]] || \
	{
	    echo "${json_config}: missing" |& tee -a current.log > /dev/stderr
	    echo "Did you run ./prepare.sh?" |& tee -a current.log
	    exit 2
	}

    for f in nn_tiny nn nn_big nn_bigger; do
	onnx_input=/project/bfys/$USER/codebaby/Allen/input/ghost_${f}.onnx
	[[ -f ${onnx_input} ]] || \
	    {
		echo "${onnx_input}: missing" |& tee -a current.log > /dev/stderr
		exit 3
	    }

	echo "config: ${json_config}" |& tee -a current.log
	echo "onnx: ${onnx_input}" |& tee -a current.log
	python ./scanprops.py ${json_config} \
	       --experiment-name v100 \
	       --batch-size-range 256 16000 \
	       --fp16 \
	       --int8 \
	       --no-infer \
	       --onnx-input ${onnx_input} \
	       --copies $i |& tee -a current.log
    done
done

echo "config: hlt1_pp_default_seq.json" |& tee -a current.log
python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
	   --experiment-name v100 |& tee -a current.log
