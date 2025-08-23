export LD_LIBRARY_PATH=$CONDA_LOC/pkgs/mkl-2025.0.0-hacee8c2_941/lib:$LD_LIBRARY_PATH
export MKL_NUM_THREADS=24      # all cores
export OMP_NUM_THREADS=24
export KMP_AFFINITY=granularity=fine,compact,1,0