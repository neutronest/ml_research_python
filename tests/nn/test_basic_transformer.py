import numpy as np
import torch
import torch.nn as nn

from tests.nn import test_constants
from research.nn import helper
from research.nn.basic_transformer import BasicEncoder, BasicEncoderLayer
from research.nn.basic_transformer import BasicDecoder, BasicDecoderLayer
from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork

def test_basic_encoder():
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_INPUT)
    
    multihead_attention = NaiveMultiHeadSelfAttention(
        test_constants.N_HEAD,
        test_constants.N_INPUT,
        test_constants.N_INPUT,
        test_constants.N_INPUT,
        test_constants.N_HIDDEN
    )
    feedforward_attention = NaiveFeedForwardNeuralNetwork(
        test_constants.N_HEAD,
        test_constants.N_INPUT,
        test_constants.N_HIDDEN
    )
    
    basic_encoder_layer_machine = BasicEncoderLayer(
        test_constants.N_INPUT,
        multihead_attention,
        feedforward_attention,
        dropout_prob=0.5
    )

    