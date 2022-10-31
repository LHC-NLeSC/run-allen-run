#!/bin/bash

# set -o xtrace

declare builddir="$1"
pushd "$builddir"

declare branch=$(git branch --show-current) sequence

[[ -n $branch ]] || \
    {
	echo "aborting, working directory in detached head"
	exit
    }

if [[ $branch == master ]]; then
    sequence=hlt1_pp_default
else
    sequence=ghostbuster_test
fi

./toolchain/wrapper ./Allen -t 12 --events-per-slice 1000 -n 1000 -r 100 \
		    --write-configuration 1 --sequence $sequence \
		    --mdf /data/bfys/raaij/upgrade/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf
popd
