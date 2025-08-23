#!/bin/bash
# Directory for logs
mkdir -p logs

# Number of repetitions (rp_time)
rp_time=10000

# Sweep matrix sizes: 512, 1024, ..., 8192
for ((n=512; n<=8192; n*=2))
do
    log_file="logs/gemv_${n}x${n}.log"
    echo "Running GEMV for ${n}x${n}..."
    ./main $n $n $rp_time |& tee $log_file
done
