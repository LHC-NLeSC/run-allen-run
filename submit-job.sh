#!/bin/bash

# exclusive node
#PBS -n
#PBS -l nodes=1:v100
#PBS -l walltime=36:00:00
#PBS -l mem=4gb
#PBS -t 1-6%2

pushd /project/bfys/$USER/codebaby/run-allen-run || exit

declare LOG=current-${PBS_ARRAYID}.log
echo "Job: ${PBS_JOBID}" |& tee $LOG

source lhcb-setup.sh
source venv/bin/activate

mlflow experiments search 2>/dev/null | grep -q v100 || \
    mlflow experiments create -n v100

[[ -d Allen-ghostbuster/build ]] || \
    {
	echo "No build of ghostbuster branch found" |& tee -a $LOG > /dev/stderr
	echo "Did you run ./build.sh & ./prepare.sh?" |& tee -a $LOG
	exit 1
    }

echo "Jobs w/ ghostbuster:" |& tee -a $LOG
cp -v ./allen_benchmarks.py Allen-ghostbuster/configuration/python/AllenCore/ |& tee -a $LOG

    export CUDA_VISIBLE_DEVICES=$( (( $i % 2 )) )
if [[ ${PBS_ARRAYID} -lt 6 ]]; then
    json_config=Allen-ghostbuster/build/ghostbuster_test_n${PBS_ARRAYID}_seq.json
    [[ -f ${json_config} ]] || \
	{
	    echo "${json_config}: missing" |& tee -a $LOG > /dev/stderr
	    echo "Did you run ./prepare.sh?" |& tee -a $LOG
	    exit 2
	}

    for f in nn_tiny nn nn_big nn_bigger; do
	onnx_input=/project/bfys/$USER/codebaby/Allen/input/ghost_${f}.onnx
	[[ -f ${onnx_input} ]] || \
	    {
		echo "${onnx_input}: missing" |& tee -a $LOG > /dev/stderr
		exit 3
	    }

	echo "config: ${json_config}" |& tee -a $LOG
	echo "onnx: ${onnx_input}" |& tee -a $LOG
	python ./scanprops.py ${json_config} \
	       --experiment-name v100 \
	       --batch-size-range 1000 16000 \
	       --fp16 \
	       --int8 \
	       --no-infer \
	       --onnx-input ${onnx_input} \
	       --copies ${PBS_ARRAYID} |& tee -a $LOG
    done
else  # ghostbusterhc_test
    export CUDA_VISIBLE_DEVICES=0
    json_config=Allen-ghostbuster/build/ghostbusterhc_test_seq.json
    [[ -f ${json_config} ]] || \
	{
	    echo "${json_config}: missing" |& tee -a $LOG > /dev/stderr
	    echo "Did you run ./prepare.sh?" |& tee -a $LOG
	    exit 2
	}

    echo "config: ${json_config}" |& tee -a $LOG
    python ./scanprops.py ${json_config} \
	   --experiment-name v100 \
	   --batch-size-range 1000 16000 \
	   --block-dim-range 16 1024 |& tee -a $LOG
fi

# echo "config: hlt1_pp_default_seq.json" |& tee -a $LOG
# python ./scanprops.py Allen-ghostbuster/build/hlt1_pp_default_seq.json \
# 	   --experiment-name v100 |& tee -a $LOG
