from spmvcodegen import SpMVCodeGen
from resource import compute_optimum_num_ch, print_resource
from runtime_estimate import estimate_runtimes

import argparse

def main():
    parser = argparse.ArgumentParser(description="Script to Generate TAPA code for SpMV")
    
    parser.add_argument("home_directory", help="Path to home directory, that is the CodeGen Folder")
    parser.add_argument("build_directory", help="Path to build directory, can be any location, new or existing [WARNING: anything inside this directory will be erased]")
    parser.add_argument("--num-ch-A", type=int, help="Number of HBM channels to read sparse matrix A")
    parser.add_argument("--num-ch-x", type=int, help="Number of HBM channels to read dense vector x")
    parser.add_argument("--num-ch-y", type=int, help="Number of HBM channels to read/write dense vector y_in/y_out")
    parser.add_argument("--ch-width", type=int, help="Width of HBM channels")
    
    args = parser.parse_args()
    
    home_dir = args.home_directory
    build_dir = args.build_directory

    num_ch_A = args.num_ch_A
    num_ch_B = args.num_ch_x
    num_ch_C = args.num_ch_y
    ch_width = args.ch_width
    assert(num_ch_A%(2*num_ch_C) == 0)
    assert(ch_width == 256 or ch_width == 512)

    myGen = SpMVCodeGen(num_ch_A, num_ch_B, num_ch_C, ch_width, build_dir, home_dir)
    myGen.generateAll()

if __name__ == "__main__":
    main()