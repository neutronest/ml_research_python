from __future__ import print_function
import torch
import torch.nn as nn

from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork

n_batch = 2
n_head = 3
n_seq = 5
embedding_size = 7
n_hidden = 11


def test_query_weight_shape():

    multi_head_self_attention_machine = NaiveMultiHeadSelfAttention(
        embedding_size,
        n_hidden,
        n_head
    )
    input_x = torch.randn(n_batch, n_seq, embedding_size)
    query_vector = multi_head_self_attention_machine.w_queries(input_x)
    assert(query_vector.shape \
        == torch.Size([n_batch, n_seq, n_head * n_hidden]))

def test_naive_multi_head_self_attention_forward():
    multi_head_self_attention_machine = NaiveMultiHeadSelfAttention(
        embedding_size,
        n_hidden,
        n_head
    )
    input_x = torch.randn(n_batch, n_seq, embedding_size)
    output = multi_head_self_attention_machine.forward(input_x)
    assert(output.size() == torch.Size([n_batch, n_seq, n_head*n_hidden]))

def test_naive_feedforward_neural_network():
    multi_head_self_attention_machine = NaiveMultiHeadSelfAttention(
        embedding_size,
        n_hidden,
        n_head
    )
    naive_feedforward_neural_network_machine = NaiveFeedForwardNeuralNetwork(
        n_head,
        n_hidden,
        n_hidden
    )

    input_x = torch.randn(n_batch, n_seq, embedding_size)
    attention_output = multi_head_self_attention_machine.forward(input_x)
    output = naive_feedforward_neural_network_machine.forward(attention_output)
    print(output.size())

