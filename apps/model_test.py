import torch
import torch.nn as nn
import numpy as np
import argparse

from pyhispmv import FpgaHandle
from fpga_layer_manager import FpgaLayerManager

import time
import os
from pathlib import Path

from model import ThreeLayerFCModel, ThreeLayerFCModelConfig, compare_model_outputs

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
    parser.add_argument('--density1', type=float, default=0.1, help="Density for the first sparse layer")
    parser.add_argument('--density2', type=float, default=0.25, help="Density for the second sparse layer")
    
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    
    # Create model config from command-line arguments
    config = ThreeLayerFCModelConfig(
        input_size=args.input_size,
        dense_size=args.hidden_size_1,
        sparse_size1=args.hidden_size_2,
        sparse_size2=args.output_size,
        density1=args.density1,
        density2=args.density2
    )
    
    cpu_model = ThreeLayerFCModel(config)
    cpu_model.eval()
    for param in cpu_model.parameters():
        param.requires_grad = False
    
    random_input = torch.randn((args.batch_size, args.input_size))  # Tokenized input for GPT-2

    fpga_model = FpgaLayerManager().replace_layers(cpu_model, fpga)

    fpga_model.eval()
    start_time = time.time()
    with torch.no_grad():
        fpga_output = fpga_model(random_input)  
    fpga_inference_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        cpu_output = cpu_model(random_input)
    cpu_inference_time = time.time() - start_time
    
    print("\n")
    print(f"FPGA Inference Time: {fpga_inference_time:.6f} seconds")
    print(f"CPU Inference Time (single thread): {cpu_inference_time:.6f} seconds")
    
    compare_model_outputs(cpu_output, fpga_output)
    
if __name__ == "__main__":
    main()