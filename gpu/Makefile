all: spmv

spmv: 
	nvcc  ./src/* -o spmv -lnvidia-ml -lcuda -lcusparse -lpthread  -I"./include"

clean:
	rm -f spmv
