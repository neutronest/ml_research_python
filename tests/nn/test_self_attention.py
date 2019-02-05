from __future__ import print_function
import torch
import torch.nn as nn

from research.nn.self_attention import NaiveMultiHeadSelfAttention

batch_size = 2
n_head = 3
seq_length = 5
embedding_size = 7
hidden_size = 11


def test_query_weight_shape():

    multi_head_self_attention_machine = NaiveMultiHeadSelfAttention(
        embedding_size,
        hidden_size,
        n_head
    )
    input_x = torch.randn(batch_size, seq_length, embedding_size)
    query_vector = multi_head_self_attention_machine.w_queries(input_x)
    assert(query_vector.shape \
        == torch.Size([batch_size, seq_length, n_head * hidden_size]))
    
   

def test_naive_multi_head_self_attention_forward():
    multi_head_self_attention_machine = NaiveMultiHeadSelfAttention(
        embedding_size,
        hidden_size,
        n_head
    )
    input_x = torch.randn(batch_size, seq_length, embedding_size)
    output = multi_head_self_attention_machine.forward(input_x)
    assert(output.size() == torch.Size([batch_size, n_head, seq_length, hidden_size]))
