rm -rf logs
mkdir logs
rm -rf power_logs
mkdir power_logs
for file in ../matrices/*/*.mtx
do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    log_file="logs/${filename}.log"
    ./spmv $file 60 |& tee $log_file
done