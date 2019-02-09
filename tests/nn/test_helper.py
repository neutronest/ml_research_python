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

def test_scaled_dot_product_attention():
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT)
    n_query = test_constants.N_INPUT
    n_head = test_constants.N_HEAD
    n_hidden = test_constants.N_HIDDEN
    w_queries = nn.Linear(n_query, n_head * n_hidden)
    w_keyes = nn.Linear(n_query, n_head * n_hidden)
    w_values = nn.Linear(n_query, n_head * n_hidden)

    nn.init.normal_(w_queries.weight, mean=0, std=np.sqrt(2.0 / (n_query+n_hidden)))
    nn.init.normal_(w_keyes.weight, mean=0, std=np.sqrt(2.0 / (n_query+n_hidden)))
    nn.init.normal_(w_values.weight, mean=0, std=np.sqrt(2.0 / (n_query+n_hidden)))

    query_input = w_queries(input_x)
    key_input = w_keyes(input_x)
    value_input = w_values(input_x)
    attention_output, attention_scores = helper.scaled_dot_product_attention(
        query_input,
        key_input,
        value_input
    )
    assert (attention_output.size() == torch.Size([
        test_constants.N_BATCH,
        test_constants.N_SEQ,
        test_constants.N_HEAD * test_constants.N_HIDDEN]))
    
    assert( attention_scores.size() == torch.Size([
        test_constants.N_BATCH,
        test_constants.N_SEQ,
        test_constants.N_SEQ
    ]))