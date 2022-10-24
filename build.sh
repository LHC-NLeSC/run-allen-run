#!/bin/bash

declare ${TOOLCHAIN:=/cvmfs/lhcb.cern.ch/lib/lhcb/lcg-toolchains/LCG_101/x86_64-centos7-clang12+cuda11_4-opt.cmake}

declare repo="$1" branch="$2"

# # FIXME: make it work with remote repo
# tip=$(GIT_DIR="$repo/.git" git show-ref --hash=9 "refs/heads/$branch")
# cdate=$(git log --date=short --format=%ad -1 "$branch")
srcdir="Allen-${branch}"
git clone "$repo" -b $branch --depth 20 "$srcdir"

declare threads=$(( $(grep --color=never -m1 'cpu cores' /proc/cpuinfo | sed -ne 's/.\+: \([0-9]\+\)/\1/p') / 2 ))
builddir="$srcdir/build"
mkdir -p "$builddir" && \
    {
	pushd "$builddir"
	cmake -DSTANDALONE=ON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN ../
	make -j"$threads" |& tee build.log
	popd
    }
