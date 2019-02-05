import torch
import torch.nn as nn

from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork


class NaiveEncoder(nn.Module):
    def __init__(
        self,
        n_head,
        n_input,
        n_hidden):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.mutli_head_attention = NaiveMultiHeadSelfAttention(
            n_input,
            n_hidden,
            n_head
        )
        self.feed_forward = NaiveFeedForwardNeuralNetwork(
            n_head,
            n_hidden,
            n_hidden
        )
        return
    
    def forward(self, input_query, input_key, input_value):
        attention_output = self.mutli_head_attention(input_query, input_key, input_value)
        output = self.feed_forward(attention_output)
        return output
    



class NaiveTransformer:
    def __init__(
        self, 
        n_layer,
        n_head,
        n_input,
        n_hidden):
        return
