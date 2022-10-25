source /cvmfs/lhcb.cern.ch/lib/LbEnv

# CPU: -DCMAKE_TOOLCHAIN_FILE=/cvmfs/lhcb.cern.ch/lib/lhcb/lcg-toolchains/LCG_101/x86_64-centos7-clang12-opt.cmake
# CUDA: -DCMAKE_TOOLCHAIN_FILE=/cvmfs/lhcb.cern.ch/lib/lhcb/lcg-toolchains/LCG_101/x86_64-centos7-clang12+cuda11_4-opt.cmake

export CUDNN_ROOT=/project/bfys/$USER/build/cuda
export TRT_ROOT=/project/bfys/$USER/build/TensorRT-8.4.0.6
export UMESIMD_ROOT_DIR=/project/bfys/$USER/build
