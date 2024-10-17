from commons import SpMVConfig
from spmvcodegen import SpMVCodeGen
from resource import ResourceEstimator
from fpgas import U280

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
    parser.add_argument("--urams-per-pe", type=int, help="URAM banks allocated to store output for each PE")
    parser.add_argument("--dense-overlay", action="store_true", help="Build SpMV kernel with support for GeMV")
    parser.add_argument("--pre-accumulator", action="store_true", help="Build SpMV kernel with Pre-Accumulator")
    parser.add_argument("--row-dist-net", action="store_true", help="Build SpMV kernel with Row Distribution Network")
    args = parser.parse_args()
    
    home_dir = args.home_directory
    build_dir = args.build_directory
    
    mySpMV = SpMVConfig(
        num_ch_A = args.num_ch_A,
        num_ch_B = args.num_ch_x,
        num_ch_C = args.num_ch_y,
        ch_width = args.ch_width,
        urams_per_pe = args.urams_per_pe,
        dense_overlay = args.dense_overlay,
        pre_accumulator = args.pre_accumulator,
        row_dist_net = args.row_dist_net
    )

    assert(mySpMV.num_ch_A%(2*mySpMV.num_ch_C) == 0)
    assert(mySpMV.ch_width == 256 or mySpMV.ch_width == 512)

    print("Configuration: ", mySpMV)

    myGen = SpMVCodeGen(mySpMV, build_dir, home_dir)
    myGen.generateAll()

    myEst = ResourceEstimator.getEstimateFromConfig(mySpMV, U280)
    print("Resource Estimate: ", myEst)

if __name__ == "__main__":
    main()