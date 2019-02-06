import numpy as np
import torch
import torch.nn as nn

def clones(module, n_layers):
    """
    Produce n layers for module
    """
    return nn.MoudleList([copy.deepcopy(moduole) for _ in range(N)])
