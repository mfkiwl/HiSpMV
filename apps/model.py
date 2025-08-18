import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from sparse_dot_mkl import dot_product_mkl

import time

# Generic Sparse Layer with Bias
class SparseLinear(nn.Module):
    def __init__(self, input_size, output_size, density):
        super(SparseLinear, self).__init__() 
        self.input_size = input_size
        self.output_size = output_size
        self.density = density
        self.weight = self._initialize_sparse_weights()
        # Convert weight to SciPy CSR format for MKL
        coo = self.weight.coalesce()  # Ensure indices are sorted and unique
        self.csr_weight = sp.csr_matrix((coo.values().numpy(), (coo.indices()[0].numpy(), coo.indices()[1].numpy())), shape=coo.shape)
        self.bias = nn.Parameter(torch.zeros(output_size)) 

    def _initialize_sparse_weights(self):
        size = (self.output_size, self.input_size)
        dense_weight = torch.randn(size)
        mask = torch.rand_like(dense_weight) < self.density
        sparse_weight_dense = dense_weight * mask 
        sparse_weight = sparse_weight_dense.to_sparse()
        num_nonzero = sparse_weight._values().numel()
        total_elements = torch.prod(torch.tensor(size)).item()
        density = num_nonzero / total_elements
        print("Density:", density)
        return sparse_weight

    def forward(self, x):
        # PyTorch sparse.mm Impl.
        # output = torch.sparse.mm(self.weight, x.T).T  # Sparse matrix multiplication
        # output += self.bias  # Add the bias
        # MKL dot product impl
        x_np = x.detach().cpu().numpy()  # Convert to NumPy
        output_np = np.broadcast_to(self.bias.cpu().numpy(), (x_np.shape[0], self.bias.shape[0]))  # Ensure batch dimension
        output_np = np.ascontiguousarray(output_np).T
        output_np = dot_product_mkl(self.csr_weight, x_np.T).T
        output = torch.from_numpy(output_np).to(x.device)
        return output

# Configuration class to manage model parameters
class ThreeLayerFCModelConfig:
    def __init__(self, input_size, dense_size, sparse_size1, sparse_size2, density1, density2):
        self.input_size = input_size
        self.dense_size = dense_size
        self.sparse_size1 = sparse_size1
        self.sparse_size2 = sparse_size2
        self.density1 = density1
        self.density2 = density2

# Three-Layer Fully Connected Model
class ThreeLayerFCModel(nn.Module):
    def __init__(self, config: ThreeLayerFCModelConfig, rp_time=100):
        super(ThreeLayerFCModel, self).__init__()
        self.config = config
        # Using config for model parameters
        self.dense = nn.Linear(config.input_size, config.dense_size)
        self.sparse1 = SparseLinear(config.dense_size, config.sparse_size1, config.density1)
        self.sparse2 = SparseLinear(config.sparse_size1, config.sparse_size2, config.density2)
        self.activation = nn.ReLU()
        self.rp_time = rp_time

    def forward(self, x):
        start = time.time()
        for i in range(self.rp_time):
            x1 = self.activation(self.dense(x))  # Dense layer expects input shape (batch_size, input_size)
        layer1 = time.time()
        for i in range(self.rp_time):
            x2 = self.activation(self.sparse1(x1))  # Sparse layer also expects input shape (batch_size, dense_size)
        layer2 = time.time()
        for i in range(self.rp_time):
            x3 = self.activation(self.sparse2(x2))  # Sparse layer expects (batch_size, sparse_size1)
        layer3 = time.time()
        print((layer1-start)/self.rp_time, (layer2-layer1)/self.rp_time, (layer3-layer2)/self.rp_time)
        return x3
    
def compare_model_outputs(ref_output, cmp_output):
    """
    Compares the output of two models (e.g., CPU and FPGA) and prints:
    - Absolute error histogram
    - Relative error histogram
    - Maximum absolute and relative errors
    - Values at the indices with max errors

    Args:
    ref_output (torch.Tensor): Output from the ref model.
    cmp_output (torch.Tensor): Output from the compare model.
    """
    # Flatten the outputs
    ref_output = ref_output.flatten()
    cmp_output = cmp_output.flatten()

    # Calculate the absolute and relative errors
    abs_error = torch.abs(ref_output - cmp_output)
    rel_error = torch.abs((ref_output - cmp_output) / (ref_output + 1e-8))

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
    print(f"Ref Output: {ref_output[max_abs_error_idx].item()}, Actual Output: {cmp_output[max_abs_error_idx].item()}")

    print(f"Values at Max Relative Error (index {max_rel_error_idx.item()}):")
    print(f"Ref Output: {ref_output[max_rel_error_idx].item()}, Actual Output: {cmp_output[max_rel_error_idx].item()}")