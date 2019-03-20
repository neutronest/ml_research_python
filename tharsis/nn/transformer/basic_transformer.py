from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from tharsis.nn.helper import LayerNorm, ShortConnectionLayer, clones
from tharsis.nn.transformer.basic_self_attention import PositionwiseFeedForward
from tharsis.nn.transformer.basic_self_attention import BasicMultiHeadSelfAttention



class BasicEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        """
        layer: nn.Moudle subclass 
               the layer module (self-attention, etc.)
        n_layers: int
               number of layers
        """
        super(BasicEncoder, self).__init__()
        self.n_layers = n_layers
        self.norm = LayerNorm(layer.n_features)
        self.layers = clones(layer, n_layers)
        return
    
    def forward(self, input_x, mask=None):
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
        super(BasicDecoder, self).__init__()
        self.n_layers = n_layers
        self.norm = LayerNorm(layer.n_features)
        self.layers = clones(layer, n_layers)
        return
    
    def forward(self, input_x, memory, source_mask=None, target_mask=None):
        output = input_x
        for layer in self.layers:
            output = layer(output, memory, source_mask, target_mask)
        return self.norm(output)

class BasicDecoderLayer(nn.Module):
    def __init__(
        self,
        n_features,
        self_attention_layer,
        source_attention_layer,
        feedforward_layer,
        dropout_prob=0.5):
        """
        """
        super(BasicDecoderLayer, self).__init__()
        self.n_features = n_features
        self.self_attention_layer = self_attention_layer
        self.source_attention_layer = source_attention_layer
        self.feedforward_layer = feedforward_layer
        self.dropout_prob = dropout_prob
        self.sublayers = clones(ShortConnectionLayer(n_features, dropout_prob), 3)
        return
    
    def forward(self, input_x, encoder_memory, source_mask=None, target_mask=None):
        memory = encoder_memory
        output = input_x
        output = self.sublayers[0](output, lambda x: self.self_attention_layer(x, x, x, target_mask))
        output = self.sublayers[1](output, lambda x: self.source_attention_layer(x, memory, memory, source_mask))
        return self.sublayers[2](output, self.feedforward_layer)

"""
From http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

"""
From http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
class Embeddings(nn.Module):
    def __init__(self, n_vocab, n_hidden):
        super(Embeddings, self).__init__()
        self.embedding_fn = nn.Embedding(n_vocab, n_hidden)
        self.n_hidden = n_hidden

    def forward(self, input_x):
        # todo: why?
        return self.embedding_fn(input_x) * math.sqrt(self.n_hidden)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, n_input, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(n_input, vocab)

    def forward(self, input_x):
        return nn.functinoal.log_softmax(self.proj(input_x), dim=-1)

class Classifier(nn.Module):
    def __init__(self, n_input, n_labels):
        super(Generator, self).__init__()
        self.proj = nn.Linear(n_input, n_labels)
    def forward(self, input_x):
        return nn.functinoal.log_softmax(self.proj(input_x), dim=-1)


class BasicTransformer(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        n_vocab,
        n_hidden,
        dropout_prob):
        super(BasicTransformer, self).__init__()
        
        attention_machine = BasicMultiHeadSelfAttention(
            n_head=n_head,
            n_hidden=n_hidden,
            dropout_prob=dropout_prob
        )
        feedforward_machine = PositionwiseFeedForward(
            n_input=n_hidden,
            n_hidden=n_hidden
        )
        
        self.encoder_layer = BasicEncoderLayer(
            n_features=n_hidden,
            self_attention_layer=deepcopy(attention_machine),
            feedforward_layer=deepcopy(feedforward_machine),
            dropout_prob=dropout_prob
        )
        self.encoder = BasicEncoder(
            self.encoder_layer, 
            n_layer)
        
        self.decoder_layer = BasicDecoderLayer(
            n_features=n_hidden,
            self_attention_layer=deepcopy(attention_machine),
            source_attention_layer=deepcopy(attention_machine),
            feedforward_layer=deepcopy(feedforward_machine),
            dropout_prob=dropout_prob
        )
        self.decoder = BasicDecoder(
            self.decoder_layer,
            n_layer
        )
        self.init()
    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
    
    def forward(self, input_x):
        encoder_output = self.encoder(input_x)
        decoder_output = self.decoder(input_x, encoder_output)
        return decoder_output
