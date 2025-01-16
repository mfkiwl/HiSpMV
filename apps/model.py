import torch
import torch.nn as nn

# Generic Sparse Layer with Bias
class SparseLinear(nn.Module):
    def __init__(self, input_size, output_size, sparsity=0.1):
        super(SparseLinear, self).__init__() 
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity

        self.weight = self._initialize_sparse_weights()
        self.bias = nn.Parameter(torch.zeros(output_size)) 

    def _initialize_sparse_weights(self):
        dense_weight = torch.randn(self.output_size, self.input_size)
        mask = torch.rand_like(dense_weight) > self.sparsity
        sparse_weight_dense = dense_weight * mask 
        sparse_weight = sparse_weight_dense.to_sparse()
        return sparse_weight

    def forward(self, x):
        # Perform sparse matrix multiplication and add bias
        output = torch.sparse.mm(self.weight, x.T).T  # Sparse matrix multiplication
        output += self.bias  # Add the bias
        return output

# Configuration class to manage model parameters
class ThreeLayerFCModelConfig:
    def __init__(self, input_size, dense_size, sparse_size1, sparse_size2, sparsity1=0.1, sparsity2=0.01):
        self.input_size = input_size
        self.dense_size = dense_size
        self.sparse_size1 = sparse_size1
        self.sparse_size2 = sparse_size2
        self.sparsity1 = sparsity1
        self.sparsity2 = sparsity2

# Three-Layer Fully Connected Model
class ThreeLayerFCModel(nn.Module):
    def __init__(self, config: ThreeLayerFCModelConfig):
        super(ThreeLayerFCModel, self).__init__()
        self.config = config
        # Using config for model parameters
        self.dense = nn.Linear(config.input_size, config.dense_size)
        self.sparse1 = SparseLinear(config.dense_size, config.sparse_size1, sparsity=config.sparsity1)
        self.sparse2 = SparseLinear(config.sparse_size1, config.sparse_size2, sparsity=config.sparsity2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.dense(x))  # Dense layer expects input shape (batch_size, input_size)
        x = self.activation(self.sparse1(x))  # Sparse layer also expects input shape (batch_size, dense_size)
        x = self.activation(self.sparse2(x))  # Sparse layer expects (batch_size, sparse_size1)
        return x