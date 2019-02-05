from __future__ import print_function
import torch
import torch.nn as nn

from research.nn.self_attention import MultiHeadSelfAttention

def test_query_weight_shape():

    sequence_size = 4
    hidden_size = 8
    embedding_size = 10
    n_head = 2
    batch_size = 10
    multi_head_self_attention_machine = MultiHeadSelfAttention(
        embedding_size,
        hidden_size,
        n_head
    )
    input_x = torch.randn(batch_size, sequence_size, embedding_size)
    # assert(multi_head_self_attention_machine.w_queries(input_x).shape \
    #     == torch.Size([batch_size, n_head * hidden_size]))
    
    multi_head_self_attention_machine.forward(input_x)