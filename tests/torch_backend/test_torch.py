from __future__ import print_function
import torch
import torch.nn as nn

def test_nn_linear():
    n_input = 20
    n_hidden = 30
    batch_size = 10
    mlp_layer = nn.Linear(n_input, n_hidden)
    input_matrix = torch.randn(batch_size, n_input)
    output_matrix = mlp_layer(input_matrix)
    assert(output_matrix.size() == torch.Size([batch_size, n_hidden]))

