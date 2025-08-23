MKL_INCLUDE := "$(CONDA_LOC)/pkgs/mkl-include-2025.0.0-hc79277c_941/include"
MKL_LIB := "$(CONDA_LOC)/pkgs/mkl-2025.0.0-hacee8c2_941/lib"

all: main

main: 
	g++ -o main -O2 src/* -I$(MKL_INCLUDE) -L$(MKL_LIB) -lmkl_rt

clean:
	rm -f main
