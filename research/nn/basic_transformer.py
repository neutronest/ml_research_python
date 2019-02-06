import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        """
        layer: nn.Moudle subclass 
               the layer module (self-attention, etc.)
        n_layers: int
               number of layers
        """
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        