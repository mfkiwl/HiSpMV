import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from transformers import Conv1D

class FpgaLayerManager:
    """
    Manages FPGA operations, including handle creation and resource allocation.
    """
    def __init__(self, fpga, custom_linear_class):
        self.fpga = fpga  
        self.custom_linear_class = custom_linear_class

    def process_weights(self, layer):
        """
        Processes weights of a given layer and creates an FPGA handle.
        Determines whether to use sparse or dense representation based on density.
        """
        if not hasattr(layer, 'weight'):
            raise ValueError("Layer must have a weight attribute.")

        # Extract weights and bias
        weight = layer.weight.t().detach().numpy() if isinstance(layer, Conv1D) else layer.weight.detach().numpy()
        bias = layer.bias.detach().numpy() if hasattr(layer, 'bias') and layer.bias is not None else np.zeros(weight.shape[0], dtype=weight.dtype)

        # Calculate density
        density = np.count_nonzero(weight) / weight.size
    
        # Create dense or sparse handle based on density
        # handle_idx = 0
        if density > 0.5:
            handle_idx = self.fpga.create_dense_handle(weight.flatten(), *weight.shape)
        else:
            coo = coo_matrix(weight)
            handle_idx = self.fpga.create_sparse_handle(coo.row, coo.col, coo.data, *weight.shape)

        if handle_idx == -1:
            raise RuntimeError("FPGA memory is full.")

        return handle_idx, bias

    def replace_layers(self, model):
        """
        Replaces nn.Linear and Conv1D layers in a model with custom FPGA-enabled layers.
        """
        new_model = model.__class__(model.config) if hasattr(model, 'config') else model.__class__()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                print(name)
                handle_idx, bias = self.process_weights(module)
                layers = name.split(".")
                target_layer = new_model
                for layer in layers[:-1]:
                    target_layer = getattr(target_layer, layer)
                setattr(target_layer, layers[-1], self.custom_linear_class(self.fpga, handle_idx, bias))
        self.fpga.load_matrices()
        return new_model