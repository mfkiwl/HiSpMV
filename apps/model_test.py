import torch
import torch.nn as nn
import numpy as np
import argparse

from pyhispmv import FpgaHandle
from fpga_layer_manager import FpgaLayerManager
from fpga_linear import FpgaLinear

import time
import os
from pathlib import Path

from model import ThreeLayerFCModel, ThreeLayerFCModelConfig

# Get the current file's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parents[0]

# Initialize the FPGA handle
xclbin_path = os.path.join(parent_dir, "builds/Dense-HI-SpMV-24-1-1/SpMV_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin")
device_id = 0  # Example device ID
num_ch_A = 24
num_ch_B = 1
num_ch_C = 1
urams_per_pe = 2
fp_acc_latency = 5
dense_overlay = True
pre_accumulator = False
row_dist_net = True

# Create an FpgaHandle object
fpga = FpgaHandle(
    xclbin_path, device_id, num_ch_A, num_ch_B, num_ch_C, 
    urams_per_pe, fp_acc_latency, dense_overlay, pre_accumulator, row_dist_net
)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Three-Layer Fully Connected Model with CPU and FPGA inference.")
    
    # Model configuration arguments
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--input_size', type=int, default=4096, help="Input size")
    parser.add_argument('--hidden_size_1', type=int, default=8192, help="Size of the first hidden layer")
    parser.add_argument('--hidden_size_2', type=int, default=8192, help="Size of the second hidden layer")
    parser.add_argument('--output_size', type=int, default=1024, help="Output size")
    parser.add_argument('--sparsity1', type=float, default=0.01, help="Sparsity for the first sparse layer")
    parser.add_argument('--sparsity2', type=float, default=0.1, help="Sparsity for the second sparse layer")
    
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    
    # Create model config from command-line arguments
    config = ThreeLayerFCModelConfig(
        input_size=args.input_size,
        dense_size=args.hidden_size_1,
        sparse_size1=args.hidden_size_2,
        sparse_size2=args.output_size,
        sparsity1=args.sparsity1,
        sparsity2=args.sparsity2
    )
    cpu_model = ThreeLayerFCModel(config)

    for param in cpu_model.parameters():
        param.requires_grad = False
    
    random_input = torch.randn((args.batch_size, args.input_size))  # Tokenized input for GPT-2

    cpu_model.eval()
    # Perform inference and measure time
    start_time = time.time()
    with torch.no_grad():
        cpu_output = cpu_model(random_input)
    cpu_inference_time = time.time() - start_time


    fpga_model = FpgaLayerManager(fpga, FpgaLinear).replace_layers(cpu_model)

    fpga_model.eval()
    start_time = time.time()
    with torch.no_grad():
        fpga_output = fpga_model(random_input)  
    fpga_inference_time = time.time() - start_time

    print("\n")
    print(f"CPU Inference Time: {cpu_inference_time:.6f} seconds")
    print(f"FPGA Inference Time: {fpga_inference_time:.6f} seconds")
    # Max absolute error
    # Calculate max absolute error
    cpu_output = cpu_output.flatten()
    fpga_output = fpga_output.flatten()

    # Calculate the absolute and relative errors
    abs_error = torch.abs(cpu_output - fpga_output)
    rel_error = torch.abs((cpu_output - fpga_output) / (cpu_output + 1e-8))

    # Convert errors to numpy arrays for easy processing
    abs_error_np = abs_error.cpu().numpy()
    rel_error_np = rel_error.cpu().numpy()

    # Define the bin edges for histograms
    bins = np.linspace(0, max(np.max(abs_error_np), np.max(rel_error_np)), 11)  # 10 bins

    # Calculate the histogram for absolute error
    abs_hist, abs_bin_edges = np.histogram(abs_error_np, bins=bins)

    # Calculate the histogram for relative error
    rel_hist, rel_bin_edges = np.histogram(rel_error_np, bins=bins)

    # Print the absolute error histogram (range and count)
    print("\nAbsolute Error Histogram:")
    for i in range(len(abs_hist)):
        print(f"Range: ({abs_bin_edges[i]:.4f}, {abs_bin_edges[i+1]:.4f}), Count: {abs_hist[i]}")

    # Print the relative error histogram (range and count)
    print("\nRelative Error Histogram:")
    for i in range(len(rel_hist)):
        print(f"Range: ({rel_bin_edges[i]:.4f}, {rel_bin_edges[i+1]:.4f}), Count: {rel_hist[i]}")

    # Calculate max absolute and relative errors
    max_abs_error = torch.max(abs_error).item()
    max_rel_error = torch.max(rel_error).item()

    # Find the index of max absolute and relative errors
    max_abs_error_idx = torch.argmax(abs_error)
    max_rel_error_idx = torch.argmax(rel_error)

    # Print the errors and values at the max error indices
    print("\n")
    print(f"Max Absolute Error: {max_abs_error}")
    print(f"Max Relative Error: {max_rel_error}")

    print(f"Values at Max Absolute Error (index {max_abs_error_idx.item()}):")
    print(f"CPU Output: {cpu_output[max_abs_error_idx].item()}, FPGA Output: {fpga_output[max_abs_error_idx].item()}")

    print(f"Values at Max Relative Error (index {max_rel_error_idx.item()}):")
    print(f"CPU Output: {cpu_output[max_rel_error_idx].item()}, FPGA Output: {fpga_output[max_rel_error_idx].item()}")

    
if __name__ == "__main__":
    main()