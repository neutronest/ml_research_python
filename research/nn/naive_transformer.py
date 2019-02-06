import torch
import torch.nn as nn

from research.nn.self_attention import NaiveMultiHeadSelfAttention
from research.nn.self_attention import NaiveFeedForwardNeuralNetwork
from research.nn.self_attention import PositionwiseFeedForwardNetwork

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
        # self.feed_forward = NaiveFeedForwardNeuralNetwork(
        #     n_head,
        #     n_hidden,
        #     n_hidden
        # )
        self.feed_forward = PositionwiseFeedForwardNetwork(n_head*n_hidden, n_head*n_query)
        return
    
    def forward(self, input_query, input_key, input_value):
        attention_output = self.mutli_head_attention(input_query, input_key, input_value)
        output = self.feed_forward(attention_output)
        return output
    

class NaiveDecoder(nn.Module):
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
        
        self.decode_multihead_attention_fn = NaiveMultiHeadSelfAttention(
            n_head=n_head,
            n_query=n_query,
            n_key=n_key,
            n_value=n_value,
            n_hidden=n_hidden
        )
        self.encode_decode_multihead_attention_fn = NaiveMultiHeadSelfAttention(
            n_head=n_head,
            n_query=n_hidden,
            n_key=n_hidden,
            n_value=n_hidden,
            n_hidden=n_hidden
        )
        self.feed_forward = NaiveFeedForwardNeuralNetwork(
            n_head,
            n_hidden,
            n_hidden
        )
        return
    
    def forward(self, decode_input_query, encode_output):

        """
        shape: n_batch, n_seq, n_head * n_hidden
        encoder_output
        """
        decode_attention_output = self.decode_multihead_attention_fn(
            decode_input_query, decode_input_query, decode_input_query
        )
        import pdb
        pdb.set_trace()

        encode_decode_attention_output = self.encode_decode_multihead_attention_fn(
            decode_attention_output, encode_output, encode_output
        )
        output = self.feed_forward(encode_decode_attention_output)
        return output

class NaiveTransformer:
    def __init__(
        self, 
        n_layer,
        n_head,
        n_input,
        n_hidden):
        return
