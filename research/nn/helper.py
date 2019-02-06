import numpy as np
import torch
import torch.nn as nn

def clones(module, n_layers):
    """
    Produce n layers for module
    """
    return nn.MoudleList([copy.deepcopy(moduole) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    employ a residual connection 
    always around each of the two sub-layers
    cite: https://arxiv.org/pdf/1512.03385.pdf 
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.activation = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, input_x):
        """
        input_x: torch.Tensor
        """
        mean = input_x.mean(-1, keepdim=True)
        std = input_x.std(-1, keepdim=True)
        result = self.activation * (input_x - mean) / (std + self.eps) + self.bias
        return result
