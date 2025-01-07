import torch
import torch.nn as nn
import numpy as np

from pyhispmv import FpgaHandle
from fpga_layer_manager import FpgaLayerManager
from fpga_linear import FpgaLinear

import time
import os
from pathlib import Path

from transformers import GPT2LMHeadModel

# Get the current file's directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parents[0]

# Limit PyTorch to a single thread
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def main():
    # Initialize the GPT-2 XL language model head
    cpu_model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

    # # Load pruned model weights directly
    pruned_weight_file = os.path.join(parent_dir, "pruning/fc_pruned_gpt2xl_weights.pth")
    checkpoint = torch.load(pruned_weight_file, map_location=torch.device('cpu'), weights_only=True)
    cpu_model.load_state_dict(checkpoint, strict=False)

    # Freeze all the parameters (weights and biases)
    for param in cpu_model.parameters():
        param.requires_grad = False
    
    random_input = torch.randint(0, 50257, (1, 128), dtype=torch.long)  # Tokenized input for GPT-2

    cpu_model.eval()

    print("STARTING CPU SOON...")
    time.sleep(1)
    # Perform inference and measure time
    start_time = time.time()
    with torch.no_grad():
        cpu_output = cpu_model(random_input).logits
    cpu_inference_time = time.time() - start_time
    print("\n")
    print(f"CPU Inference Time: {cpu_inference_time:.6f} seconds")


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
    fpga_model = FpgaLayerManager(fpga, FpgaLinear).replace_layers(cpu_model)
    # fpga_model = measure_execution_time(cpu_model)
    
    for param in fpga_model.parameters():
        param.requires_grad = False

    fpga_model.eval()

    print("STARTING FPGA SOON...")
    time.sleep(1)
    start_time = time.time()
    with torch.no_grad():
        fpga_output = fpga_model(random_input).logits
    fpga_inference_time = time.time() - start_time

    print(f"FPGA Inference Time: {fpga_inference_time:.6f} seconds")
    # Max absolute error
    # Calculate max absolute error
    cpu_output = cpu_output.flatten()
    fpga_output = fpga_output.flatten()
    max_abs_error = torch.max(torch.abs(cpu_output - fpga_output)).item()

    # Calculate max relative error
    max_rel_error = torch.max(torch.abs((cpu_output - fpga_output) / (cpu_output))).item()

    # Find the index of max absolute error
    max_abs_error_idx = torch.argmax(torch.abs(cpu_output - fpga_output))

    # Find the index of max relative error
    max_rel_error_idx = torch.argmax(torch.abs((cpu_output - fpga_output) / (cpu_output)))

    # Print errors
    print("\n")
    print(f"Max Absolute Error: {max_abs_error}")
    print(f"Max Relative Error: {max_rel_error}")

    # Print the values at the max error index
    print(f"Values at Max Absolute Error (index {max_abs_error_idx.item()}):")
    print(f"CPU Output: {cpu_output[max_abs_error_idx].item()}, FPGA Output: {fpga_output[max_abs_error_idx].item()}")

    print(f"Values at Max Relative Error (index {max_rel_error_idx.item()}):")
    print(f"CPU Output: {cpu_output[max_rel_error_idx].item()}, FPGA Output: {fpga_output[max_rel_error_idx].item()}")

    
if __name__ == "__main__":
    main()