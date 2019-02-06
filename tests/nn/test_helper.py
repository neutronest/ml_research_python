import numpy as np
import torch
import torch.nn as nn

from tests.nn import test_constants
from research.nn import helper

def test_layer_norm():
    
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT)
    
    layer_norm_machine = helper.LayerNorm(test_constants.N_INPUT)
    output = layer_norm_machine(input_x)
    assert(output.size() ==  torch.Size([
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT]))

def test_short_connection():
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT)
    sublayer = helper.LayerNorm(test_constants.N_INPUT)
    short_connection_machine = helper.ShortConnectionLayer(
        n_features=test_constants.N_INPUT,
        dropout_prob=0.5
    )
    output = short_connection_machine(input_x, sublayer)
    assert(output.size() == torch.Size([
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT]))