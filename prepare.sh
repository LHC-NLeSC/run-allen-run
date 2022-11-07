#!/bin/bash

# set -o xtrace

declare builddir="$1"
pushd "$builddir"

declare branch=$(git branch --show-current)

[[ -n $branch ]] || \
    {
	echo "aborting, working directory in detached head"
	exit
    }

function write_seq_2_json() {
    local sequence=$1
    ./toolchain/wrapper ./Allen -t 1 --events-per-slice 1000 -n 1000 -r 100 \
			--write-configuration 1 --sequence $sequence \
			--mdf /data/bfys/raaij/upgrade/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf
    mv Sequence.json ${sequence}_seq.json
}

if [[ $branch == master ]]; then
    write_seq_2_json hlt1_pp_default
else
    write_seq_2_json hlt1_pp_default
    write_seq_2_json ghostbuster_test
fi

popd
