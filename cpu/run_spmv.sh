for file in ../matrices/*/*.mtx
do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    log_file="logs/${filename}.log"
    ./main $file 200 |& tee $log_file
done