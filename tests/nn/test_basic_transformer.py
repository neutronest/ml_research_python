import numpy as np
import torch
import torch.nn as nn

from tests.nn import test_constants
from research.nn import helper
from research.nn.basic_transformer import BasicEncoder, BasicEncoderLayer
from research.nn.basic_transformer import BasicDecoder, BasicDecoderLayer
from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.basic_self_attention import BasicMultiHeadSelfAttention
from research.nn.basic_self_attention import PositionwiseFeedForward
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork

def test_basic_encoder():
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT)
    
    multihead_attention = BasicMultiHeadSelfAttention(
        test_constants.N_BATCH,
        test_constants.N_INPUT
    )
    feedforward_attention = PositionwiseFeedForward(
        test_constants.N_INPUT,
        test_constants.N_HIDDEN
    )
    
    basic_encoder_layer_machine = BasicEncoderLayer(
        test_constants.N_INPUT,
        multihead_attention,
        feedforward_attention,
        dropout_prob=0.5
    )
    output = basic_encoder_layer_machine(input_x)