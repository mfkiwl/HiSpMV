import numpy as np
import torch
import torch.nn as nn


class FpgaLinear(nn.Module):
    """
    A PyTorch custom linear layer that offloads computation to an FPGA.
    """
    def __init__(self, fpga, matrix_idx, bias):
        super(FpgaLinear, self).__init__()
        self.fpga = fpga  # Instance of FPGA handle class
        self.matrix_idx = matrix_idx  # Matrix index in FPGA memory
        self.bias_npy = bias # Bias

    def forward(self, x):
       y = self.fpga.linear(self.matrix_idx, x.view(-1).numpy(), self.bias_npy)
        return torch.from_numpy(y).view(*x.shape[:-1], self.bias_npy.shape[0])