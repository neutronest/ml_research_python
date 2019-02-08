import copy
import numpy as np
import torch
import torch.nn as nn

def clones(module, n_layers):
    """
    Produce n layers for module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class LayerNorm(nn.Module):
    """
    employ a residual connection 
    always around each of the two sub-layers
    cite: https://arxiv.org/pdf/1512.03385.pdf 
    """
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.activation = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
    
    def forward(self, input_x):
        """
        input_x: torch.Tensor
        """
        mean = input_x.mean(-1, keepdim=True)
        std = input_x.std(-1, keepdim=True)
        result = self.activation * (input_x - mean) / (std + self.eps) + self.bias
        return result

class ShortConnectionLayer(nn.Module):
    """
    """
    def __init__(self, n_features, dropout_prob):
        super(ShortConnectionLayer, self).__init__()
        self.norm = LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_x, sublayer):
        """
        input_x: torch.Tensor
        sublayer: nn.Module subclass
        """
        return input_x + self.dropout(sublayer(self.norm(input_x)))
    

def scaled_dot_product_attention(
    query_input,
    key_input,
    value_input,
    mask=None,
    dropout_prob=None):
    """
    query_input's dimension = key_input
    """
    # can use like that???
    n_query = query_input.size(-1)
    temprature = np.power(n_query, 0.5)
    query_key_scores = torch.matmul(query_input, key_input.transpose(-2, -1)) / temprature
    if mask is not None:
        # todo: ?
        query_key_scores = query_key_scores.mask_fill(mask==0, -1e9)
    softmax_scores = nn.functional.softmax(query_key_scores, -1)
    if dropout_prob is not None:
        softmax_fn = nn.Dropout(dropout_prob)
        softmax_scores = softmax_fn(softmax_scores)
    return torch.matmul(softmax_scores, value_input), softmax_scores