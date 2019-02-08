import numpy as np
import torch
import torch.nn as nn

from research.nn.helper import LayerNorm, ShortConnectionLayer, clones
from research.nn.basic_self_attention import PositionwiseFeedForward


class BasicEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        """
        layer: nn.Moudle subclass 
               the layer module (self-attention, etc.)
        n_layers: int
               number of layers
        """
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.norm = LayerNorm(layer.n_features)
        self.layers = clones(layer, n_layers)
        return
    
    def forward(self, input_x, mask):
        output = input_x
        for layer in self.layers:
            output = layer(output, mask)
        return self.norm(output)

class BasicEncoderLayer(nn.Module):
    def __init__(
        self,
        n_features,
        self_attention_layer, 
        feedforward_layer, 
        dropout_prob=0.5):
        """
        TODO: n_features for what?
        """
        super(BasicEncoderLayer, self).__init__()
        self.n_features = n_features
        self.self_attention_layer = self_attention_layer
        self.feedforward_layer = feedforward_layer
        self.dropout_prob=0.5

        self.sublayers = clones(ShortConnectionLayer(n_features, dropout_prob), 2)
        return
    
    def forward(self, input_x, mask=None):
        """
        TODO: mask type???
        """
        def attention_layer_fn(input_x):
            return self.self_attention_layer(input_x, input_x, input_x, mask)
        def feedforward_fn(input_x):
            return self.feedforward_layer(input_x)
        
        attention_output = self.sublayers[0].forward(input_x, attention_layer_fn)
        feedforward_output = self.sublayers[1].forward(attention_output, feedforward_fn)
        return feedforward_output


class BasicDecoder(nn.Module):
    def __init__(self, layer, n_layers):
        """
        layer: nn.Moudle subclass 
               the layer module (self-attention, etc.)
        n_layers: int
               number of layers
        """
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.norm = LayerNorm(layer.n_features)
        self.layers = clones(layer, n_layers)
        return
    
    def forward(self, input_x, memory, source_mask, target_mask):
        output = input_x
        for layer in self.layers:
            output = layer(output, memory, source_mask, target_mask)
        return self.norm(output)

class BasicDecoderLayer(nn.Module):
    def __init__(
        self, 
        self_attention_layer,
        source_attention_layer,
        feedforward_layer,
        dropout_prob=0.5):
        """
        """
        super(BasicDecoderLayer, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.source_attention_layer = source_attention_layer
        self.feedforward_layer = feedforward_layer
        self.dropout_prob = dropout_prob
        self.sublayers = clones(ShortConnectionLayer(n_features, dropout_prob), 3)
        return
    
    def forward(self, input_x, encoder_memory, source_mask, target_mask):
        memory = encoder_memory
        output = input_x
        output = self.sublayers[0](output, lambda x: self.self_attention_layer(x, x, x, target_mask))
        output = self.sublayers[1](output, lambda x: self.source_attention_layer(x, memory, memory, source_mask))
        return self.sublayers[2](output, self.feedforward_layer)
        