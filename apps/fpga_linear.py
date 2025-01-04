import numpy as np
import torch
import torch.nn as nn

class FpgaLinear(nn.Module):
    """
    A PyTorch custom linear layer that offloads computation to an FPGA.
    """
    def __init__(self, fpga, matrix_idx, bias_npy):
        super(FpgaLinear, self).__init__()
        self.fpga = fpga  # Instance of FPGA handle class
        self.matrix_idx = matrix_idx  # Matrix index in FPGA memory
        self.bias_npy = bias_npy  # Bias as a NumPy array

    def forward(self, x):
        # Flatten all but the last dimension
        x_npy = x.numpy()
        y_npy = np.empty_like(self.bias_npy)
        self.fpga.select_matrix(self.matrix_idx)
        self.fpga.run_kernel(x_npy, self.bias_npy, y_npy, 1.0, 1.0)
        return torch.from_numpy(y_npy)
        # Flatten all but the last dimension
        # x_flat = x.view(-1, x.shape[-1])  # Torch tensor (no conversion to NumPy yet)
        
        # Pre-allocate output tensor
        # y = torch.empty((*x.shape[:-1], self.bias_npy.shape[0]), dtype=x.dtype)
        
        
        # # Select the matrix and run kernels on the FPGA
        # self.fpga.select_matrix(self.matrix_idx)
        # for i in range(x_flat.shape[0]):
        #     # Convert input and bias to NumPy, compute kernel output directly in NumPy
        #     x_npy = x_flat[i].numpy()
        #     y_npy = np.empty_like(self.bias_npy)
        #     # self.fpga.run_kernel(x_npy, self.bias_npy, y_npy, 1.0, 1.0)
        #     y_flat[i] = torch.from_numpy(y_npy)  # Avoid reshaping NumPy arrays here
        
        # # Reshape output to match the input dimensions
        # y = y_flat.view(*x.shape[:-1], self.bias_npy.shape[0])
        # return y

class FpgaConv1D(nn.Module):
    """
    A PyTorch custo Conv1D layer that offloads computation to an FPGA.
    """
    def __init__(self, fpga, matrix_idx, bias_npy):
        super(FpgaConv1D, self).__init__()
        self.fpga = fpga  # Instance of FPGA handle class
        self.matrix_idx = matrix_idx  # Matrix index in FPGA memory
        self.bias_npy = bias_npy  # Bias as a NumPy array

    def forward(self, x):
        # Flatten all but the last dimension
        x_npy = x.numpy()
        y_npy = np.empty_like(self.bias_npy)
        self.fpga.select_matrix(self.matrix_idx)
        self.fpga.run_kernel(x_npy, self.bias_npy, y_npy, 1.0, 1.0)
        return torch.from_numpy(y_npy)
        # x_npy = x.view(-1, x.shape[-1]).numpy()
        # y_npy = np.empty((x_npy.shape[0], self.bias_npy.shape[0]), dtype=np.float32)

        # # Select the matrix and run kernels on the FPGA
        # self.fpga.select_matrix(self.matrix_idx)
        # for i in range(x_npy.shape[0]):
        #     self.fpga.run_kernel(x_npy[i], self.bias_npy, y_npy[i], 1.0, 1.0)

        # # Reshape output back to match the input dimensions
        # y_reshaped = y_npy.reshape(*x.shape[:-1], self.bias_npy.shape[0])
        # return torch.from_numpy(y_reshaped)