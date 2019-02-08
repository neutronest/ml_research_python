import numpy as np
import torch
import torch.nn as nn

from research.nn import helper


class BasicMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        n_head,
        n_hidden,
        dropout_prob=0.5):
        super(BasicMultiHeadSelfAttention, self).__init__()
        self.n_head = n_head

        self.linear_fns = helper.clones(nn.Linear(n_hidden, n_hidden), 4)
        self.dropout_prob = dropout_prob
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # todo: ?
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        query_encode = query
        key_encode = key
        value_encode = value
        query_encode, key_encode, value_encode = [
            l(x).view(n_batch, -1, self.n_head, self.n_hidden).transpose(1, 2)
            for l, x in zip(self.linear_fns, (query_encode, key_encode, value_encode))
        ]
        value_attention_output, attention = helper.scaled_dot_product_attention(
            query_encode, key_encode, value_encode, dropout_prob=self.dropout_prob)
        output = value_attention_output.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.n_hidden )
        return self.linear_fns[-1](output)





