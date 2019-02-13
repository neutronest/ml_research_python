import numpy as np
import torch
import torch.nn as nn

from tests.nn import test_constants
from research.nn import helper
from research.nn.transformer.basic_transformer import BasicEncoder, BasicEncoderLayer
from research.nn.transformer.basic_transformer import BasicDecoder, BasicDecoderLayer
from research.nn.transformer.basic_transformer import BasicTransformer
from research.nn.transformer.basic_transformer import Embeddings
from research.nn.transformer.basic_self_attention import BasicMultiHeadSelfAttention
from research.nn.transformer.basic_self_attention import PositionwiseFeedForward
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork
from research.nn.self_attention import NaiveMultiHeadSelfAttention


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

def test_basic_transformer():
    input_x = torch.randn(
        test_constants.N_BATCH, 
        test_constants.N_SEQ,
        test_constants.N_HIDDEN)
    
    basic_transformer_machine = BasicTransformer(
        n_layer=1,
        n_head=test_constants.N_HEAD,
        n_vocab=test_constants.N_INPUT,
        n_hidden=test_constants.N_HIDDEN,
        dropout_prob=0.5
    )

    # embedding_machine = Embeddings(
    #     n_vocab=test_constants.N_INPUT,
    #     n_hidden=test_constants.N_HIDDEN
    # )

    # embedding_x = embedding_machine(input_x)
    decoder_output = basic_transformer_machine(input_x)
    print(decoder_output.size())