import torch
import torch.nn as nn

from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork


class NaiveEncoder(nn.Module):
    def __init__(
        self,
        n_head,
        n_query,
        n_key,
        n_value,
        n_hidden):
        super().__init__()
        self.n_head = n_head
        self.n_query = n_query
        self.n_key = n_key
        self.n_value = n_value
        self.n_hidden = n_hidden

        self.mutli_head_attention = NaiveMultiHeadSelfAttention(
            n_head=n_head,
            n_query=n_query,
            n_key=n_key,
            n_value=n_value,
            n_hidden=n_hidden
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
    

class NaiveDecoder(nn.Module):
    def __init__(
        self,
        n_head,
        n_input,
        n_hidden,):
        super().__init__()
        self.n_head = n_head
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.decoder_multihead_attention = NaiveMultiHeadSelfAttention(
            n_input,
            n_hidden,
            n_head
        )


        return
    
    def forward(self, decoder_input_query, encoder_input_key, encoder_input_value):

        #decoder_attention_output =
        return





class NaiveTransformer:
    def __init__(
        self, 
        n_layer,
        n_head,
        n_input,
        n_hidden):
        return
